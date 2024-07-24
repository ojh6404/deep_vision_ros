from typing import List
import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
import supervision as sv


from tracking_ros.model_base import InferenceConfigBase, InferenceModelBase
from tracking_ros.utils import overlay_davis, nhw_to_hw, hw_to_nhw

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
        return detections.xyxy, labels, scores, visualization


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
                boxes, labels, scores, _ = detection_model.predict(image)
                detections = sv.Detections(
                    xyxy=boxes,
                    mask=None,
                    class_id=np.array([detection_model.classes.index(label) for label in labels]),
                    confidence=np.array(scores),
                )

                # segment objects with SAM
                result_masks = []
                segmentation, _ = sam_model.predict(image=image, boxes=boxes)
                result_masks = np.array(hw_to_nhw(segmentation))

                detections.mask = np.array(result_masks)
                incorporate_mask = torch.zeros(
                    image.shape[:2], dtype=torch.int64, device=self.model_config.device
                )
                curr_id = 1
                segments_info = []
                # sort by descending area to preserve the smallest object
                for i in np.flip(np.argsort(detections.area)):
                    mask = detections.mask[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    mask = torch.from_numpy(mask.astype(np.float32))
                    mask = (mask > 0.5).float()
                    if mask.sum() > 0:
                        incorporate_mask[mask > 0] = curr_id
                        segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence))
                        curr_id += 1
                prob = self.predictor.incorporate_detection(torch_image, incorporate_mask, segments_info)
                self.cnt = 1
            else:  # just track objects
                prob = self.predictor.step(torch_image, None, None)
                self.cnt += 1
            segmentation = torch.argmax(prob, dim=0).cpu().numpy().astype(np.int32)  # (H, W)
            object_num = len(np.unique(segmentation)) - 1  # without 0 (background)
            if object_num > 0:
                masks = []
                object_ids = np.unique(segmentation)[1:]  # without 0 (background)
                for i in object_ids:
                    mask = (segmentation == i).astype(np.uint8)
                    masks.append(mask)
                masks = np.stack(masks, axis=0)  # (N, H, W)
                xyxy = sv.mask_to_xyxy(masks)
                detections = sv.Detections(
                    xyxy=xyxy,
                    mask=masks,
                    class_id=object_ids,
                )
                painted_image = overlay_davis(image.copy(), segmentation)
                # TODO convert labels to class name, but it needs some trick because object id and class id is not consistent between tracking and detecting
                visualization = BOX_ANNOTATOR.annotate(
                    scene=painted_image,
                    detections=detections,
                    # labels=[f"ObjectID: {obj_id}" for obj_id in object_ids],
                )

                # label_names = [self.classes[cls_id] for cls_id in detections.class_id]
                # label_array = LabelArray()
                # label_array.labels = [Label(id=i + 1, name=name) for i, name in enumerate(label_names)]
                # label_array.header.stamp = rospy.Time.now()
                # label_array.header.frame_id = img_msg.header.frame_id
                # class_result = ClassificationResult(
                #     header=label_array.header,
                #     classifier=self.gd_config.model_name,
                #     target_names=self.classes,
                #     labels=[self.classes.index(name) for name in label_names],
                #     label_names=label_names,
                #     label_proba=detections.confidence.tolist(),
                # )
                # self.pub_labels.publish(label_array)
                # self.pub_class.publish(class_result)
            else:
                visualization = overlay_davis(image.copy(), segmentation)
                detections = sv.Detections(
                    xyxy=np.empty((0, 4)),
                    mask=np.empty((0, 0, 0)),
                    class_id=np.empty(0),
                )
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
            raise ValueError(f"Unknown model type: {self.config.model_type}")

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
            raise ValueError(f"Unknown model type: {self.config.model_type}")

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
