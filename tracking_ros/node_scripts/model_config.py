from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import rospy
import rospkg
import torch
from typing import List

CHECKPOINT_ROOT = os.path.join(rospkg.RosPack().get_path("tracking_ros"), "trained_data")


@dataclass
class ROSInferenceModelConfig(ABC):
    model_name: str
    device: str = "cuda:0"

    # mypy doesn't understand abstractclassmethod, so we use this workaround
    @abstractmethod
    def get_predictor(self):
        pass

    @classmethod
    @abstractmethod
    def from_rosparam(cls):
        pass

    @classmethod
    @abstractmethod
    def from_args(cls):
        pass


@dataclass
class SAMConfig(ROSInferenceModelConfig):
    model_type: str = "vit_t"
    mode: str = "prompt"

    model_checkpoint_root = os.path.join(CHECKPOINT_ROOT, "sam")
    model_checkpoints = {
        "vit_t": os.path.join(model_checkpoint_root, "mobile_sam.pth"),
        "vit_b": os.path.join(model_checkpoint_root, "sam_vit_b.pth"),
        "vit_l": os.path.join(model_checkpoint_root, "sam_vit_l.pth"),
        "vit_h": os.path.join(model_checkpoint_root, "sam_vit_h.pth"),
        "vit_b_hq": os.path.join(model_checkpoint_root, "sam_vit_b_hq.pth"),
        "vit_l_hq": os.path.join(model_checkpoint_root, "sam_vit_l_hq.pth"),
        "vit_h_hq": os.path.join(model_checkpoint_root, "sam_vit_h_hq.pth"),
    }

    def get_predictor(self):
        assert self.model_type in SAMConfig.model_checkpoints
        assert self.mode in ["prompt", "automatic"]
        if "hq" in self.model_type:
            from segment_anything_hq import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        elif self.model_type == "vit_t":
            from mobile_sam import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        else:
            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        model = sam_model_registry[self.model_type[:5]](checkpoint=self.model_checkpoints[self.model_type])
        model.to(device=self.device).eval()
        return SamPredictor(model) if self.mode == "prompt" else SamAutomaticMaskGenerator(model)

    @classmethod
    def from_args(cls, model_type: str = "vit_t", mode: str = "prompt", device: str = "cuda:0"):
        return cls(model_name="SAM", model_type=model_type, mode=mode, device=device)

    @classmethod
    def from_rosparam(cls):
        return cls.from_args(
            rospy.get_param("~model_type", "vit_t"),
            rospy.get_param("~mode", "prompt"),
            rospy.get_param("~device", "cuda:0"),
        )


@dataclass
class CutieConfig(ROSInferenceModelConfig):
    model_checkpoint = os.path.join(CHECKPOINT_ROOT, "cutie/cutie-base-mega.pth")

    def get_predictor(self):
        from omegaconf import open_dict
        import hydra
        from hydra import compose, initialize

        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with torch.inference_mode():
            initialize(
                version_base="1.3.2",
                config_path="../Cutie/cutie/config",
                job_name="eval_config",
            )
            cfg = compose(config_name="eval_config")

            with open_dict(cfg):
                cfg["weights"] = self.model_checkpoint
            data_cfg = get_dataset_cfg(cfg)

            cutie = CUTIE(cfg).to(self.device).eval()
            model_weights = torch.load(cfg.weights, map_location=self.device)
            cutie.load_weights(model_weights)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return InferenceCore(cutie, cfg=cfg)

    @classmethod
    def from_args(cls, device: str = "cuda:0"):
        return cls(model_name="Cutie", device=device)

    @classmethod
    def from_rosparam(cls):
        return cls.from_args(rospy.get_param("~device", "cuda:0"))


@dataclass
class DEVAConfig(ROSInferenceModelConfig):
    model_checkpoint = os.path.join(CHECKPOINT_ROOT, "deva/DEVA-propagation.pth")

    def get_predictor(self):
        from argparse import ArgumentParser
        from deva.model.network import DEVA
        from deva.inference.inference_core import DEVAInferenceCore
        from deva.inference.eval_args import add_common_eval_args
        from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args

        # default parameters
        parser = ArgumentParser()
        add_common_eval_args(parser)
        add_ext_eval_args(parser)
        add_text_default_args(parser)
        args = parser.parse_args([])

        # deva model
        args.model = self.model_checkpoint

        cfg = vars(args)
        cfg["enable_long_term"] = True

        # Load our checkpoint
        deva_model = DEVA(cfg).to(self.device).eval()
        if args.model is not None:
            model_weights = torch.load(args.model, map_location=self.device)
            deva_model.load_weights(model_weights)
        else:
            print("No model loaded.")

        # TODO clean it and make it configurable
        cfg["enable_long_term_count_usage"] = True
        cfg["max_num_objects"] = 50
        cfg["amp"] = True
        cfg["chunk_size"] = 4
        cfg["detection_every"] = 5
        cfg["max_missed_detection_count"] = 10
        cfg["temporal_setting"] = "online"
        cfg["pluralize"] = True
        cfg["DINO_THRESHOLD"] = 0.5

        deva = DEVAInferenceCore(deva_model, config=cfg)
        deva.next_voting_frame = cfg["num_voting_frames"] - 1
        deva.enabled_long_id()

        return deva, cfg

    @classmethod
    def from_args(cls, device: str = "cuda:0"):
        return cls(model_name="DEVA", device=device)

    @classmethod
    def from_rosparam(cls):
        return cls.from_args(rospy.get_param("~device", "cuda:0"))


@dataclass
class GroundingDINOConfig(ROSInferenceModelConfig):
    model_config = os.path.join(CHECKPOINT_ROOT, "groundingdino/GroundingDINO_SwinT_OGC.py")
    model_checkpoint = os.path.join(CHECKPOINT_ROOT, "groundingdino/groundingdino_swint_ogc.pth")

    def get_predictor(self):
        try:
            from groundingdino.util.inference import Model as GroundingDINOModel
        except ImportError:
            from GroundingDINO.groundingdino.util.inference import (
                Model as GroundingDINOModel,
            )
        return GroundingDINOModel(
            model_config_path=self.model_config,
            model_checkpoint_path=self.model_checkpoint,
            device=self.device,
        )

    @classmethod
    def from_args(cls, device: str = "cuda:0"):
        return cls(model_name="GroundingDINO", device=device)

    @classmethod
    def from_rosparam(cls):
        return cls.from_args(rospy.get_param("~device", "cuda:0"))


@dataclass
class YOLOConfig(ROSInferenceModelConfig):
    model_id: str = "yolov8x-worldv2.pt"

    def get_predictor(self):
        from ultralytics import YOLOWorld

        return YOLOWorld(
            self.model_id,
        )

    @classmethod
    def from_args(cls, model_id: str = "yolov8x-worldv2.pt", device: str = "cuda:0"):
        return cls(model_name="YOLO", model_id=model_id, device=device)

    @classmethod
    def from_rosparam(cls):
        return cls.from_args(
            rospy.get_param("~model_id", "yolov8x-worldv2.pt"),
            rospy.get_param("~device", "cuda:0"),
        )


@dataclass
class VLPartConfig(ROSInferenceModelConfig):
    model_type: str = "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed"
    model_root = rospkg.RosPack().get_path("tracking_ros") + "/VLPart"
    model_checkpoint_root = os.path.join(CHECKPOINT_ROOT, "vlpart")
    model_checkpoints = {
        "r50_voc": os.path.join(model_checkpoint_root, "r50_voc.pth"),
        "r50_coco": os.path.join(model_checkpoint_root, "r50_coco.pth"),
        "r50_lvis": os.path.join(model_checkpoint_root, "r50_lvis.pth"),
        "r50_partimagenet": os.path.join(model_checkpoint_root, "r50_partimagenet.pth"),
        "r50_pascalpart": os.path.join(model_checkpoint_root, "r50_pascalpart.pth"),
        "r50_paco": os.path.join(model_checkpoint_root, "r50_paco.pth"),
        "r50_lvis_paco": os.path.join(model_checkpoint_root, "r50_lvis_paco.pth"),
        "r50_lvis_paco_pascalpart": os.path.join(model_checkpoint_root, "r50_lvis_paco_pascalpart.pth"),
        "r50_lvis_paco_pascalpart_partimagenet": os.path.join(
            model_checkpoint_root, "r50_lvis_paco_pascalpart_partimagenet.pth"
        ),
        "r50_lvis_paco_pascalpart_partimagenet_in": os.path.join(
            model_checkpoint_root, "r50_lvis_paco_pascalpart_partimagenet_in.pth"
        ),
        "r50_lvis_paco_pascalpart_partimagenet_inparsed": os.path.join(
            model_checkpoint_root, "r50_lvis_paco_pascalpart_partimagenet_inparsed.pth"
        ),
        "swinbase_cascade_voc": os.path.join(model_checkpoint_root, "swinbase_cascade_voc.pth"),
        "swinbase_cascade_coco": os.path.join(model_checkpoint_root, "swinbase_cascade_coco.pth"),
        "swinbase_cascade_lvis": os.path.join(model_checkpoint_root, "swinbase_cascade_lvis.pth"),
        "swinbase_cascade_partimagenet": os.path.join(model_checkpoint_root, "swinbase_cascade_partimagenet.pth"),
        "swinbase_cascade_pascalpart": os.path.join(model_checkpoint_root, "swinbase_cascade_pascalpart.pth"),
        "swinbase_cascade_paco": os.path.join(model_checkpoint_root, "swinbase_cascade_paco.pth"),
        "swinbase_cascade_lvis_paco": os.path.join(model_checkpoint_root, "swinbase_cascade_lvis_paco.pth"),
        "swinbase_cascade_lvis_paco_pascalpart": os.path.join(
            model_checkpoint_root, "swinbase_cascade_lvis_paco_pascalpart.pth"
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet": os.path.join(
            model_checkpoint_root,
            "swinbase_cascade_lvis_paco_pascalpart_partimagenet.pth",
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in": os.path.join(
            model_checkpoint_root,
            "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.pth",
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed": os.path.join(
            model_checkpoint_root,
            "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth",
        ),
    }
    model_configs = {
        "r50_voc": os.path.join(model_root, "configs/voc/r50_voc.yaml"),
        "r50_coco": os.path.join(model_root, "configs/coco/r50_coco.yaml"),
        "r50_lvis": os.path.join(model_root, "configs/lvis/r50_lvis.yaml"),
        "r50_partimagenet": os.path.join(model_root, "configs/partimagenet/r50_partimagenet.yaml"),
        "r50_pascalpart": os.path.join(model_root, "configs/pascalpart/r50_pascalpart.yaml"),
        "r50_paco": os.path.join(model_root, "configs/joint/r50_lvis_paco.yaml"),
        "r50_lvis_paco": os.path.join(model_root, "configs/joint/r50_lvis_paco.yaml"),
        "r50_lvis_paco_pascalpart": os.path.join(model_root, "configs/joint/r50_lvis_paco_pascalpart.yaml"),
        "r50_lvis_paco_pascalpart_partimagenet": os.path.join(
            model_root, "configs/joint/r50_lvis_paco_pascalpart_partimagenet.yaml"
        ),
        "r50_lvis_paco_pascalpart_partimagenet_in": os.path.join(
            model_root, "configs/joint_in/r50_lvis_paco_pascalpart_partimagenet_in.yaml"
        ),
        "r50_lvis_paco_pascalpart_partimagenet_inparsed": os.path.join(
            model_root,
            "configs/joint_in/r50_lvis_paco_pascalpart_partimagenet_inparsed.yaml",
        ),
        "swinbase_cascade_voc": os.path.join(model_root, "configs/voc/swinbase_cascade_voc.yaml"),
        "swinbase_cascade_coco": os.path.join(model_root, "configs/coco/swinbase_cascade_coco.yaml"),
        "swinbase_cascade_lvis": os.path.join(model_root, "configs/lvis/swinbase_cascade_lvis.yaml"),
        "swinbase_cascade_partimagenet": os.path.join(
            model_root, "configs/partimagenet/swinbase_cascade_partimagenet.yaml"
        ),
        "swinbase_cascade_pascalpart": os.path.join(model_root, "configs/pascalpart/swinbase_cascade_pascalpart.yaml"),
        "swinbase_cascade_paco": os.path.join(model_root, "configs/joint/swinbase_cascade_lvis_paco.yaml"),
        "swinbase_cascade_lvis_paco": os.path.join(model_root, "configs/joint/swinbase_cascade_lvis_paco.yaml"),
        "swinbase_cascade_lvis_paco_pascalpart": os.path.join(
            model_root, "configs/joint/swinbase_cascade_lvis_paco_pascalpart.yaml"
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet": os.path.join(
            model_root,
            "configs/joint/swinbase_cascade_lvis_paco_pascalpart_partimagenet.yaml",
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in": os.path.join(
            model_root,
            "configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.yaml",
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed": os.path.join(
            model_root,
            "configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml",
        ),
    }

    def get_predictor(
        self,
        vocabulary: str = "custom",
        custom_vocabulary: List[str] = [],
        confidence_threshold: float = 0.7,
    ):
        from detectron2.config import get_cfg

        import sys
        import argparse

        sys.path.insert(0, self.model_root)
        from vlpart.config import add_vlpart_config
        from demo.predictor import VisualizationDemo

        def setup_cfg(args):
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_vlpart_config(cfg)
            cfg.merge_from_file(args.config_file)
            cfg.merge_from_list(args.opts)
            # Set score_threshold for builtin models
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

            # replace the filename in the list to the full path
            for idx, filename in enumerate(cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH_GROUP):
                if filename:
                    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH_GROUP[idx] = os.path.join(self.model_root, filename)
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH = os.path.join(
                self.model_root, cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH
            )
            for idx, filename in enumerate(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH_GROUP):
                if filename:
                    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH_GROUP[idx] = os.path.join(self.model_root, filename)
            cfg.freeze()
            return cfg

        custom_vocabulary = ",".join(custom_vocabulary) if vocabulary == "custom" else ""  # type: ignore
        args = {
            "config_file": self.model_configs[self.model_type],
            "vocabulary": vocabulary,
            "custom_vocabulary": custom_vocabulary,
            "confidence_threshold": confidence_threshold,
            "opts": [
                "MODEL.WEIGHTS",
                self.model_checkpoints[self.model_type],
                "VIS.BOX",
                "False",
            ],
        }
        print(args)

        args = argparse.Namespace(**args)  # type: ignore
        cfg = setup_cfg(args)

        demo = VisualizationDemo(cfg, args)
        return demo

    @classmethod
    def from_args(cls, device: str = "cuda:0"):
        return cls(model_name="VLPart", device=device)

    @classmethod
    def from_rosparam(cls):
        return cls.from_args(rospy.get_param("~device", "cuda:0"))


@dataclass
class MaskDINOConfig(ROSInferenceModelConfig):
    model_type: str = "panoptic_swinl"
    model_root = rospkg.RosPack().get_path("tracking_ros") + "/MaskDINO"
    model_checkpoint_root = os.path.join(CHECKPOINT_ROOT, "MaskDINO")
    model_checkpoints = {
        "instance_r50_hid1024": os.path.join(
            model_checkpoint_root,
            "instance/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth",
        ),
        "instance_r50": os.path.join(
            model_checkpoint_root,
            "instance/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth",
        ),
        "instance_swinl_no_mask_enhanced": os.path.join(
            model_checkpoint_root,
            "instance/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_mask52.1ap_box58.3ap.pth",
        ),
        "instance_swinl": os.path.join(
            model_checkpoint_root,
            "instance/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth",
        ),
        "panoptic_r50": os.path.join(
            model_checkpoint_root,
            "panoptic/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth",
        ),
        "panoptic_swinl": os.path.join(
            model_checkpoint_root,
            "panoptic/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth",
        ),
        "semantic_r50_ade20k": os.path.join(
            model_checkpoint_root,
            "semantic/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth",
        ),
        "semantic_r50_cityscapes": os.path.join(
            model_checkpoint_root,
            "semantic/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth",
        ),
    }
    model_configs = {
        "instance_r50_hid1024": os.path.join(
            model_root,
            "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml",
        ),
        "instance_r50": os.path.join(
            model_root,
            "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
        ),
        "instance_swinl_no_mask_enhanced": os.path.join(
            model_root,
            "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        ),
        "instance_swinl": os.path.join(
            model_root,
            "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        ),
        "panoptic_r50": os.path.join(
            model_root,
            "configs/coco/panoptic-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
        ),
        "panoptic_swinl": os.path.join(
            model_root,
            "configs/coco/panoptic-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        ),
        "semantic_r50_ade20k": os.path.join(
            model_root,
            "configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml",
        ),
        "semantic_r50_cityscapes": os.path.join(
            model_root,
            "configs/cityscapes/semantic-segmentation/maskdino_R50_bs16_90k_steplr.yaml",
        ),
    }

    def get_predictor(self, confidence_threshold: float = 0.7):
        import argparse
        import sys

        sys.path.insert(0, self.model_root)
        sys.path.insert(0, os.path.join(self.model_root, "demo"))
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from maskdino import add_maskdino_config
        from predictor import VisualizationDemo

        def setup_cfg(args):
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskdino_config(cfg)
            cfg.merge_from_file(args.config_file)
            cfg.merge_from_list(args.opts)
            cfg.freeze()
            return cfg

        args = {
            "config_file": self.model_configs[self.model_type],
            "confidence_threshold": confidence_threshold,
            "opts": ["MODEL.WEIGHTS", self.model_checkpoints[self.model_type]],
        }

        args = argparse.Namespace(**args)  # type: ignore
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)
        return demo

    @classmethod
    def from_args(cls, model_type: str = "panoptic_swinl", device: str = "cuda:0"):
        return cls(model_name="MaskDINO", model_type=model_type, device=device)

    @classmethod
    def from_rosparam(cls):
        return cls.from_args(
            rospy.get_param("~model_type", "panoptic_swinl"),
            rospy.get_param("~device", "cuda:0"),
        )
