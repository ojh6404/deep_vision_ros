from typing import List, Optional
import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
import supervision as sv
from mmengine.dataset import default_collate
from mmengine.runner import autocast


from deep_vision_ros.model_base import InferenceConfigBase, InferenceModelBase
from deep_vision_ros.utils import overlay_davis, nhw_to_hw, hw_to_nhw

from deva.dataset.utils import im_normalization
from deva.inference.object_info import ObjectInfo
from segment_anything.utils.amg import remove_small_regions

BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


class GroundingDINOModel(InferenceModelBase):
    def set_model(
        self,
        classes: List[str],
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
        nms_threshold: float = 0.8,
    ):
        self.classes = classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold

    @torch.inference_mode()
    def predict(self, image: np.ndarray):
        detections = self.predictor.predict_with_classes(
            image=image,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.nms_threshold,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        labels = [self.classes[cls_id] for cls_id in detections.class_id]
        scores = detections.confidence.tolist()
        labels_with_scores = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]

        visualization = image.copy()
        visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
        visualization = LABEL_ANNOTATOR.annotate(
            scene=visualization, detections=detections, labels=labels_with_scores
        )
        return detections, visualization


class YOLOModel(InferenceModelBase):
    def set_model(self, classes: List[str], box_threshold: float = 0.5, nms_threshold: float = 0.5):
        self.classes = classes
        self.predictor.set_classes(self.classes)
        self.box_threshold = box_threshold
        self.nms_threshold = nms_threshold

    @torch.inference_mode()
    def predict(self, image: np.ndarray):
        results = self.predictor.predict(
            image, save=False, conf=self.box_threshold, iou=self.nms_threshold, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [results.names[cls_id] for cls_id in detections.class_id]
        scores = detections.confidence.tolist()
        labels_with_scores = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]

        visualization = image.copy()
        visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
        visualization = LABEL_ANNOTATOR.annotate(
            scene=visualization, detections=detections, labels=labels_with_scores
        )
        return detections.xyxy, labels, scores, visualization


class DEVAModel(InferenceModelBase):
    def set_model(self):
        # TODO make some parameters configurable
        self.cnt = 0

    @torch.inference_mode()
    def predict(self, image, sam_model, detection_model):
        with torch.cuda.amp.autocast(enabled=self.model_config.amp):
            torch_image = im_normalization(torch.from_numpy(image).permute(2, 0, 1).float() / 255).to(
                self.model_config.device
            )
            if self.cnt % self.model_config.detection_every == 0:  # object detection query
                detections, _ = detection_model.predict(image)

                # segment objects with SAM
                result_masks = []
                segmentation, _ = sam_model.predict(image=image, boxes=detections.xyxy)
                result_masks = np.array(hw_to_nhw(segmentation))

                detections.mask = np.array(result_masks)
                incorporate_mask = torch.zeros(
                    image.shape[:2], dtype=torch.int64, device=self.model_config.device
                )
                curr_id = 1
                detection_segments_info = []
                # sort by descending area to preserve the smallest object
                for i in np.flip(np.argsort(detections.area)):
                    mask = detections.mask[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    mask = torch.from_numpy(mask.astype(np.float32))
                    mask = (mask > 0.5).float()
                    if mask.sum() > 0:
                        incorporate_mask[mask > 0] = curr_id
                        detection_segments_info.append(
                            ObjectInfo(id=curr_id, category_id=class_id, score=confidence)
                        )
                        curr_id += 1
                prob = self.predictor.incorporate_detection(
                    torch_image, incorporate_mask, detection_segments_info, incremental=True
                )
                self.cnt = 1
            else:
                prob = self.predictor.step(torch_image, None, None)
                self.cnt += 1
            segmentation = torch.argmax(prob, dim=0).cpu().numpy().astype(np.int32)  # (H, W)
            object_num = len(np.unique(segmentation)) - 1  # without 0 (background)

            segments_info = self.predictor.object_manager.get_current_segments_info()
            seg_id_to_mask_id = {
                key.id: value for key, value in self.predictor.object_manager.obj_to_tmp_id.items()
            }

            # filter out segments with area 0
            for seg in segments_info:
                area = int((segmentation == seg_id_to_mask_id[seg["id"]]).sum())
                seg["area"] = area
            segments_info = [seg for seg in segments_info if seg["area"] > 0]

            if object_num > 0:
                all_masks = []
                labels = []
                all_cat_ids = []
                all_scores = []
                for seg in segments_info:
                    all_masks.append(segmentation == seg_id_to_mask_id[seg["id"]])
                    labels.append(
                        f'{detection_model.classes[seg["category_id"]]} {seg_id_to_mask_id[seg["id"]]} {seg["score"]:.2f}'
                    )
                    all_cat_ids.append(seg["category_id"])
                    all_scores.append(seg["score"])
                all_masks = np.stack(all_masks, axis=0)  # (N, H, W)
                xyxy = sv.mask_to_xyxy(all_masks)
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=np.array(all_scores),
                    class_id=np.array(all_cat_ids),
                    tracker_id=np.array([seg_id_to_mask_id[seg["id"]] for seg in segments_info]),
                )
                painted_image = overlay_davis(image.copy(), segmentation)
                visualization = BOX_ANNOTATOR.annotate(
                    scene=painted_image,
                    detections=detections,
                )
                visualization = LABEL_ANNOTATOR.annotate(
                    scene=visualization,
                    detections=detections,
                    labels=labels,
                )

            else:
                visualization = overlay_davis(image.copy(), segmentation)
                detections = sv.Detections(
                    xyxy=np.empty((0, 4)),
                    mask=np.empty((0, 0, 0)),
                    class_id=np.empty(0),
                    tracker_id=np.empty(0),
                )  # empty detections
        return detections, visualization, segmentation


class SAMModel(InferenceModelBase):
    def set_model(self, refine_mask: bool = False, area_threshold: int = 400, refine_mode: str = "holes"):
        self.refine_mask = refine_mask
        self.multimask = False  # TODO
        if self.refine_mask:
            self.area_threshold = area_threshold
            self.refine_mode = refine_mode  # "holes" or "islands"

    @torch.inference_mode()
    def predict(self, image: np.ndarray, boxes, points=None, labels=None):
        self.predictor.set_image(image)
        segmentation = None  # [H, W]
        if boxes is not None:
            for i, box in enumerate(boxes):
                mask, logit = self.process_prompt(
                    points=None,
                    labels=None,
                    bbox=np.array([box[0], box[1], box[2], box[3]]),  # xyxy
                    multimask=self.multimask,
                )  # [H, W]
                if segmentation is None:
                    segmentation = mask.astype(np.int32)
                else:
                    segmentation[mask] = i + 1
        visualization = image.copy()
        if segmentation is not None:
            visualization = overlay_davis(visualization, segmentation)
        else:
            segmentation = np.zeros_like(image, dtype=np.int32)
        return segmentation, visualization

    def process_prompt(
        self,
        points=None,
        bbox=None,
        labels=None,
        mask_input=None,
        multimask: bool = True,
    ):
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            mask_input=mask_input,  # TODO
            multimask_output=multimask,
        )  # [N, H, W], B : number of prompts, N : number of masks recommended
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)]

        if self.refine_mask:
            # refine mask using logit
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bbox,
                mask_input=logit[None, :, :],
                multimask_output=multimask,
            )
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)]
            mask, _ = remove_small_regions(mask, self.area_threshold, mode=self.refine_mode)
        return mask, logit


class MaskDINOModel(InferenceModelBase):
    def set_model(self):
        if "panoptic" in self.model_config.model_type:
            self.classes = self.predictor.metadata.stuff_classes
        elif "instance" in self.model_config.model_type:
            self.classes = self.predictor.metadata.thing_classes
        elif "semantic" in self.model_config.model_type:
            self.classes = self.predictor.metadata.stuff_classes
        else:
            raise ValueError(f"Unknown model type: {self._modelconfig.model_type}")

    @torch.inference_mode()
    def predict(self, image: np.ndarray):
        segmentation = None
        xyxys = None
        labels = None
        scores = None
        scores = None

        predictions, visualized_output = self.predictor.run_on_image(image)
        visualization = visualized_output.get_image()

        if "panoptic_seg" in predictions:  # when panoptic segmentation
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            classes = [info["category_id"] for info in segments_info]
            if classes:
                segmentation = panoptic_seg.cpu().numpy().astype(np.int32)  # [H, W]
                H, W = segmentation.shape
                unique_instances = np.unique(segmentation)
                unique_instances = unique_instances[unique_instances != 0]
                N = len(unique_instances)
                masks = np.zeros((N, H, W), dtype=bool)
                for i, instance_id in enumerate(unique_instances):
                    masks[i] = segmentation == instance_id
                xyxys = sv.mask_to_xyxy(masks)
                labels = [self.classes[cls_id] for cls_id in classes]
                scores = self.model_config.confidence_threshold * np.ones(
                    len(classes)
                )  # TODO get confidence from model
        else:  # when instance or semantic segmentation
            if "sem_seg" in predictions:
                pass  # TODO
            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                classes = instances.pred_classes.tolist()
                if classes:
                    xyxys = instances.pred_boxes.tensor.numpy()  # [N, 4]
                    labels = [self.classes[cls_id] for cls_id in classes]
                    scores = instances.scores.tolist()
                    masks = instances.pred_masks.numpy().astype(np.int32)  # [N, H, W]
                    for i, mask in enumerate(masks):
                        if segmentation is None:
                            segmentation = mask
                        else:
                            segmentation[mask] = i + 1

        return xyxys, labels, scores, segmentation, visualization


class OneFormerModel(InferenceModelBase):
    def set_model(self):
        if self.model_config.task == "panoptic":
            self.classes = self.predictor.metadata.stuff_classes
        elif self.model_config.task == "instance":
            self.classes = self.predictor.metadata.thing_classes
        elif self.model_config.task == "semantic":
            self.classes = self.predictor.metadata.stuff_classes
        else:
            raise ValueError(f"Unknown model type: {self.model_config.model_type}")

    @torch.inference_mode()
    def predict(self, image: np.ndarray):
        segmentation = None
        xyxys = None
        labels = None
        scores = None
        scores = None

        predictions, visualized_output = self.predictor.run_on_image(image, task=self.model_config.task)

        if "panoptic_seg" in predictions and self.model_config.task == "panoptic":
            visualization = visualized_output["panoptic_inference"].get_image()
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            classes = [info["category_id"] for info in segments_info]
            if classes:
                segmentation = panoptic_seg.cpu().numpy().astype(np.int32)  # [H, W]
                # masks is [N, H, W] from segmentation [H, W] to use mask_to_xyxy
                H, W = segmentation.shape
                unique_instances = np.unique(segmentation)
                unique_instances = unique_instances[unique_instances != 0]
                N = len(unique_instances)
                masks = np.zeros((N, H, W), dtype=bool)
                for i, instance_id in enumerate(unique_instances):
                    masks[i] = segmentation == instance_id
                xyxys = sv.mask_to_xyxy(masks)
                labels = [self.classes[cls_id] for cls_id in classes]
                scores = self.model_config.confidence_threshold * np.ones(
                    len(classes)
                )  # TODO get confidence from model
        else:  # when instance or semantic segmentation
            if "sem_seg" in predictions:
                pass  # TODO
            if "instances" in predictions:
                pass  # TODO
                # visualization = visualized_output["instance_inference"].get_image()
                # instances = predictions["instances"].to("cpu")
                # classes = instances.pred_classes.tolist()
                # if classes:
                #     xyxys = instances.pred_boxes.tensor.numpy()  # [N, 4]
                #     labels = [self.classes[cls_id] for cls_id in classes]
                #     scores = instances.scores.tolist()
                #     masks = instances.pred_masks.numpy().astype(np.int32)  # [N, H, W]
                #     for i, mask in enumerate(masks):
                #         if segmentation is None:
                #             segmentation = mask
                #         else:
                #             segmentation[mask] = i + 1labels
                #     )

        return xyxys, labels, scores, segmentation, visualization


class CutieModel(InferenceModelBase):
    def __init__(self, config: InferenceConfigBase):
        self.model_config = config

    @torch.inference_mode()
    def set_model(self, image: np.ndarray, mask: np.ndarray):
        self.predictor = self.model_config.get_predictor()
        # initialize the model with the mask
        with torch.cuda.amp.autocast(enabled=True):
            # initialize with the mask
            mask_torch = (
                F.one_hot(
                    torch.from_numpy(mask).long(),
                    num_classes=len(np.unique(mask)),
                )
                .permute(2, 0, 1)
                .float()
                .to(self.model_config.device)
            )
            # the background mask is not fed into the model
            self.segmentation = self.predictor.step(self.preprocess(image), mask_torch[1:], idx_mask=False)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        torch_image = (
            torch.from_numpy(image.transpose(2, 0, 1)).float().to(self.model_config.device, non_blocking=True)
            / 255
        )
        return torch_image

    @torch.inference_mode()
    def predict(self, image: np.ndarray):
        with torch.cuda.amp.autocast(enabled=True):
            prediction = self.predictor.step(self.preprocess(image))
            self.segmentation = torch.max(prediction, dim=0).indices.cpu().numpy().astype(np.int32)
            visualization = overlay_davis(image.copy(), self.segmentation)
        return self.segmentation, visualization


class VLPartModel(InferenceModelBase):
    def __init__(self, config: InferenceConfigBase):
        self.model_config = config

    def set_model(self, vocabulary: str, classes: List[str], confidence_threshold: float = 0.5):
        self.predictor = self.model_config.get_predictor(
            vocabulary=vocabulary,
            custom_vocabulary=classes,
            confidence_threshold=confidence_threshold,
        )
        self.classes = classes

    @torch.inference_mode()
    def predict(self, image: np.ndarray):
        visualization = image.copy()
        segmentation = None
        xyxys = None
        labels = None
        scores = None

        # vlpart model inference
        predictions, visualization_output = self.predictor.run_on_image(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        visualization = visualization_output.get_image()
        instances = predictions["instances"].to("cpu")
        classes = instances.pred_classes.tolist()

        if classes:  # if there are any detections
            boxes = instances.pred_boxes.tensor
            scores = instances.scores
            masks = instances.pred_masks  # [N, H, W]
            segmentation = nhw_to_hw(masks.numpy())

            labels = [self.classes[cls_id] for cls_id in classes]
            scores = scores.tolist()
            xyxys = boxes.cpu().numpy()  # [N, 4]
        return xyxys, labels, scores, segmentation, visualization


class MASAModel(InferenceModelBase):
    def set_model(
        self,
        classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.2,
        fp16: bool = False,
    ):
        import sys

        sys.path.insert(0, self.model_config.model_root)
        from masa.apis import build_test_pipeline

        self.pipeline = build_test_pipeline(
            self.predictor.cfg, with_text=False if self.model_config.model_type == "masa_r50" else True
        )
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.fp16 = fp16
        self.frame_cnt = 0
        if self.model_config.model_type == "masa_gdino":
            self.text_prompt = " . ".join(classes)

    @torch.inference_mode()
    def predict(self, image: np.ndarray, detection_model: Optional[InferenceModelBase] = None):
        # image : bgr
        if detection_model is not None and self.model_config.model_type == "masa_r50":
            boxes, labels, scores, visualization = detection_model.predict(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

            det_bboxes = torch.tensor(boxes)
            det_scores = torch.tensor(scores)
            det_bboxes = torch.cat([det_bboxes, det_scores.unsqueeze(1)], dim=1)
            label_ids = [
                detection_model.classes.index(label) for label in labels
            ]  # convert label to label_id
            det_labels = torch.tensor(label_ids)

            data = dict(
                img=[image.astype(np.float32)],
                frame_id=[self.frame_cnt],
                ori_shape=[image.shape[:2]],
                img_id=[self.frame_cnt + 1],
            )

            data = self.pipeline(data)

            # forward the model
            with torch.no_grad():
                data = default_collate([data])
                if det_bboxes is not None:
                    data["data_samples"][0].video_data_samples[0].det_bboxes = det_bboxes
                    data["data_samples"][0].video_data_samples[0].det_labels = det_labels
                with autocast(enabled=self.fp16):
                    track_result = self.predictor.test_step(data)[0]

            self.frame_cnt += 1

            if "masks" in track_result[0].pred_track_instances:
                if len(track_result[0].pred_track_instances.masks) > 0:
                    track_result[0].pred_track_instances.masks = torch.stack(
                        track_result[0].pred_track_instances.masks, dim=0
                    )
                    track_result[0].pred_track_instances.masks = (
                        track_result[0].pred_track_instances.masks.cpu().numpy()
                    )
            track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(
                torch.float32
            )

            bboxes = track_result[0].pred_track_instances.bboxes  # tensor [N, 4]
            instances_id = track_result[0].pred_track_instances.instances_id  # tensor [N]
            scores = track_result[0].pred_track_instances.scores  # tensor [N]
            label_ids = track_result[0].pred_track_instances.labels  # tensor [N]

            # filter with score
            mask = scores > self.confidence_threshold
            bboxes = bboxes[mask]
            instances_id = instances_id[mask.cpu()]
            scores = scores[mask]
            label_ids = label_ids[mask]

            # labels_with_instances = [f"{label}_{instance_id}" for label, instance_id in zip(labels.cpu().numpy(), instances_id.cpu().numpy())]
            labels_with_instances = [
                f"{detection_model.classes[label_id]}_{instance_id}"
                for label_id, instance_id in zip(label_ids.cpu().numpy(), instances_id.cpu().numpy())
            ]

            visualization = image.copy()
            detections = sv.Detections(
                xyxy=bboxes.cpu().numpy(),
                tracker_id=instances_id.cpu().numpy(),
                class_id=label_ids.cpu().numpy(),
            )
            visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
            visualization = LABEL_ANNOTATOR.annotate(
                scene=visualization, detections=detections, labels=labels_with_instances
            )

            return detections, visualization
        else:
            # unified model like "masa_gdino"
            data = dict(
                img=[image.astype(np.float32)],
                frame_id=[self.frame_cnt],
                ori_shape=[image.shape[:2]],
                img_id=[self.frame_cnt + 1],
            )
            data["text"] = [self.text_prompt]
            data["custom_entities"] = [False]

            data = self.pipeline(data)

            # forward the model
            with torch.no_grad():
                data = default_collate([data])
                # measure FPS ##
                with autocast(enabled=self.fp16):
                    track_result = self.predictor.test_step(data)[0]

            self.frame_cnt += 1

            if "masks" in track_result[0].pred_track_instances:
                if len(track_result[0].pred_track_instances.masks) > 0:
                    track_result[0].pred_track_instances.masks = torch.stack(
                        track_result[0].pred_track_instances.masks, dim=0
                    )
                    track_result[0].pred_track_instances.masks = (
                        track_result[0].pred_track_instances.masks.cpu().numpy()
                    )

            track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(
                torch.float32
            )

            bboxes = track_result[0].pred_track_instances.bboxes  # tensor [N, 4]
            instances_id = track_result[0].pred_track_instances.instances_id  # tensor [N]
            scores = track_result[0].pred_track_instances.scores  # tensor [N]
            label_ids = track_result[0].pred_track_instances.labels  # tensor [N]

            # filter with score
            mask = scores > self.confidence_threshold
            bboxes = bboxes[mask]
            instances_id = instances_id[mask.cpu()]
            scores = scores[mask]
            label_ids = label_ids[mask]

            # labels_with_instances = [f"{label}_{instance_id}" for label, instance_id in zip(labels.cpu().numpy(), instances_id.cpu().numpy())]
            # labels_with_instances = [f"{detection_model.classes[label_id]}_{instance_id}" for label_id, instance_id in zip(label_ids.cpu().numpy(), instances_id.cpu().numpy())]
            labels_with_instances = [
                f"{self.classes[label_id]}_{instance_id}"
                for label_id, instance_id in zip(label_ids.cpu().numpy(), instances_id.cpu().numpy())
            ]

            visualization = image.copy()
            detections = sv.Detections(
                xyxy=bboxes.cpu().numpy(),
                tracker_id=instances_id.cpu().numpy(),
                class_id=label_ids.cpu().numpy(),
            )
            visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
            visualization = LABEL_ANNOTATOR.annotate(
                scene=visualization, detections=detections, labels=labels_with_instances
            )

            return detections, visualization
