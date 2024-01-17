from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import functools
import rospy
import rospkg
import numpy as np

CKECKPOINT_ROOT = os.path.join(rospkg.RosPack().get_path("tracking_ros"), "trained_data")


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

    model_checkpoint_root = os.path.join(CKECKPOINT_ROOT, "sam")
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
    model_checkpoint = os.path.join(CKECKPOINT_ROOT, "cutie/cutie-base-mega.pth")

    def get_predictor(self):
        import torch
        from omegaconf import open_dict
        from hydra import compose, initialize

        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg

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
            model_weights = torch.load(cfg.weights)
            cutie.load_weights(model_weights)

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
    model_checkpoint = os.path.join(CKECKPOINT_ROOT, "deva/DEVA-propagation.pth")

    def get_predictor(self):
        from argparse import ArgumentParser
        import torch

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
            model_weights = torch.load(args.model)
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
    model_config = os.path.join(CKECKPOINT_ROOT, "groundingdino/GroundingDINO_SwinT_OGC.py")
    model_checkpoint = os.path.join(CKECKPOINT_ROOT, "groundingdino/groundingdino_swint_ogc.pth")

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
class TAPNetConfig(ROSInferenceModelConfig):
    model_checkpoint = os.path.join(CKECKPOINT_ROOT, "tapnet/causal_tapir_checkpoint.npy")

    def get_predictor(self):
        # NOTE we should append tapnet to python path cause it's not a package based system though it is not clean
        import sys

        sys.path.insert(0, rospkg.RosPack().get_path("tracking_ros"))
        import jax
        import haiku as hk
        from tapnet import tapir_model

        def load_checkpoint(checkpoint_path):
            ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
            return ckpt_state["params"], ckpt_state["state"]

        def build_online_model_init(frames, points):
            model = tapir_model.TAPIR(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)
            feature_grids = model.get_feature_grids(frames, is_training=False)
            features = model.get_query_features(
                frames,
                is_training=False,
                query_points=points,
                feature_grids=feature_grids,
            )
            return features

        def build_online_model_predict(frames, features, causal_context):
            """Compute point tracks and occlusions given frames and query points."""
            model = tapir_model.TAPIR(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)
            feature_grids = model.get_feature_grids(frames, is_training=False)
            trajectories = model.estimate_trajectories(
                frames.shape[-3:-1],
                is_training=False,
                feature_grids=feature_grids,
                query_features=features,
                query_points_in_video=None,
                query_chunk_size=64,
                causal_context=causal_context,
                get_causal_context=True,
            )
            causal_context = trajectories["causal_context"]
            del trajectories["causal_context"]
            return {k: v[-1] for k, v in trajectories.items()}, causal_context

        params, state = load_checkpoint(self.model_checkpoint)

        # jax setup
        online_init = hk.transform_with_state(build_online_model_init)
        online_init_apply = jax.jit(online_init.apply)
        online_predict = hk.transform_with_state(build_online_model_predict)
        online_predict_apply = jax.jit(online_predict.apply)
        rng = jax.random.PRNGKey(42)  # TODO: make it configurable or dont use it
        online_init_apply = functools.partial(online_init_apply, params=params, state=state, rng=rng)
        online_predict_apply = functools.partial(online_predict_apply, params=params, state=state, rng=rng)
        return online_init_apply, online_predict_apply

    @classmethod
    def from_args(cls, device: str = "cuda:0"):
        return cls(model_name="TAPNet", device=device)

    @classmethod
    def from_rosparam(cls):
        return cls.from_args(rospy.get_param("~device", "cuda:0"))
