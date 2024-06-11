#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import supervision as sv
import torch
import torchvision
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from jsk_recognition_msgs.msg import ClassificationResult, Label, LabelArray, Rect, RectArray
from jsk_topic_tools import ConnectionBasedTransport
from segment_anything.utils.amg import remove_small_regions
from sensor_msgs.msg import Image
from tracking_ros_utils.srv import SamPrompt, SamPromptRequest

from tracking_ros.cfg import GroundingDINOConfig as ServerConfig
from tracking_ros.model_config import GroundingDINOConfig, SAMConfig
from tracking_ros.utils import overlay_davis

BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


class GroundingDinoNode(ConnectionBasedTransport):
    def __init__(self):
        super(GroundingDinoNode, self).__init__()
        self.reconfigure_server = Server(ServerConfig, self.config_cb)
        self.gd_config = GroundingDINOConfig.from_rosparam()
        self.predictor = self.gd_config.get_predictor()
        self.get_mask = rospy.get_param("~get_mask", False)
        if self.get_mask:
            self.sam_config = SAMConfig.from_rosparam()
            self.sam_predictor = self.sam_config.get_predictor()
            self.refine_mask = rospy.get_param("~refine_mask", False)
            if self.refine_mask:
                self.area_threshold = rospy.get_param("~area_threshold", 400)
                self.refine_mode = rospy.get_param("~refine_mode", "holes")  # "holes" or "islands"

        self.bridge = CvBridge()
        self.pub_vis_img = self.advertise("~output/debug_image", Image, queue_size=1)
        self.pub_rects = self.advertise("~output/rects", RectArray, queue_size=1)
        self.pub_labels = self.advertise("~output/labels", LabelArray, queue_size=1)
        self.pub_class = self.advertise("~output/class", ClassificationResult, queue_size=1)
        if self.get_mask:
            self.pub_seg = self.advertise("~output/segmentation", Image, queue_size=1)

    def subscribe(self):
        self.sub_image = rospy.Subscriber(
            "~input_image",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )

    def unsubscribe(self):
        self.sub_image.unregister()

    def config_cb(self, config, level):
        self.detect_flag = False
        self.classes = [_class.strip() for _class in config.classes.split(";") if _class.strip()]
        rospy.loginfo(f"Detecting Classes: {self.classes}")
        self.box_threshold = config.box_threshold
        self.text_threshold = config.text_threshold
        self.nms_threshold = config.nms_threshold
        self.detect_flag = True
        return config

    def publish_result(self, boxes, label_names, scores, mask, vis, frame_id):
        if label_names is not None:
            label_array = LabelArray()
            label_array.labels = [Label(id=i + 1, name=name) for i, name in enumerate(label_names)]
            label_array.header.stamp = rospy.Time.now()
            label_array.header.frame_id = frame_id
            self.pub_labels.publish(label_array)

            class_result = ClassificationResult(
                header=label_array.header,
                classifier=self.gd_config.model_name,
                target_names=self.classes,
                labels=[self.classes.index(name) for name in label_names],
                label_names=label_names,
                label_proba=scores,
            )
            self.pub_class.publish(class_result)

        if boxes is not None:
            rects = []
            for box in boxes:
                rect = Rect()
                rect.x = int(box[0])  # x1
                rect.y = int(box[1])  # y1
                rect.width = int(box[2] - box[0])  # x2 - x1
                rect.height = int(box[3] - box[1])  # y2 - y1
                rects.append(rect)
            rect_array = RectArray(rects=rects)
            rect_array.header.stamp = rospy.Time.now()
            rect_array.header.frame_id = frame_id
            self.pub_rects.publish(rect_array)

        if vis is not None:
            vis_img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="rgb8")
            vis_img_msg.header.stamp = rospy.Time.now()
            vis_img_msg.header.frame_id = frame_id
            self.pub_vis_img.publish(vis_img_msg)

        if mask is not None:
            seg_msg = self.bridge.cv2_to_imgmsg(mask, encoding="32SC1")
            seg_msg.header.stamp = rospy.Time.now()
            seg_msg.header.frame_id = frame_id
            self.pub_seg.publish(seg_msg)

    def callback(self, img_msg):
        if self.detect_flag:
            self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
            detections = self.predictor.predict_with_classes(
                image=self.image,
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

            visualization = self.image.copy()
            segmentation = None
            if self.get_mask and len(detections.xyxy) > 0:
                result_mask = None
                for i, box in enumerate(detections.xyxy):
                    mask, logit = self.process_prompt(
                        points=None,
                        labels=None,
                        bbox=np.array([box[0], box[1], box[2], box[3]]),
                        multimask=False,
                    )
                    if result_mask is None:
                        result_mask = mask.astype(np.uint8)
                    else:
                        result_mask[mask] = i + 1
                visualization = self.image.copy()
                if result_mask is not None:
                    visualization = overlay_davis(visualization, result_mask)
                segmentation = result_mask.astype(np.int32)
            visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
            visualization = LABEL_ANNOTATOR.annotate(
                scene=visualization, detections=detections, labels=labels_with_scores
            )
            self.publish_result(detections.xyxy, labels, scores, segmentation, visualization, img_msg.header.frame_id)

    def process_prompt(
        self,
        points=None,
        bbox=None,
        labels=None,
        mask_input=None,
        multimask: bool = True,
    ):
        self.sam_predictor.set_image(self.image)
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            mask_input=mask_input,  # TODO
            multimask_output=multimask,
        )  # [N, H, W], B : number of prompts, N : number of masks recommended
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)]

        if self.refine_mask:
            # refine mask using logit
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bbox,
                mask_input=logit[None, :, :],
                multimask_output=multimask,
            )
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)]
            mask, _ = remove_small_regions(mask, self.area_threshold, mode=self.refine_mode)
        return mask, logit


if __name__ == "__main__":
    rospy.init_node("grounding_dino_node")
    node = GroundingDinoNode()
    rospy.spin()
