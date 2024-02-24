#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import supervision as sv
import rospy

from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import Label, LabelArray
from jsk_recognition_msgs.msg import ClassificationResult
from tracking_ros_utils.srv import SamPrompt, SamPromptRequest

from tracking_ros.cfg import GroundingDINOConfig as ServerConfig
from model_config import GroundingDINOConfig


class GroundingDinoNode(ConnectionBasedTransport):
    def __init__(self):
        super(GroundingDinoNode, self).__init__()
        self.reconfigure_server = Server(ServerConfig, self.config_cb)
        self.gd_config = GroundingDINOConfig.from_rosparam()
        self.predictor = self.gd_config.get_predictor()
        self.get_mask = rospy.get_param("~get_mask", False)

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
        self.classes = [_class.strip() for _class in config.classes.split(";")]
        self.box_threshold = config.box_threshold
        self.text_threshold = config.text_threshold
        self.nms_threshold = config.nms_threshold
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

        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        self.visualization = self.image.copy()
        self.segmentation = None
        if self.get_mask and len(detections.xyxy) > 0:
            rospy.wait_for_service("/sam_node/process_prompt")
            try:
                prompt = SamPromptRequest()
                prompt.image = img_msg
                for xyxy in detections.xyxy:
                    rect = Rect()
                    rect.x = int(xyxy[0])  # x1
                    rect.y = int(xyxy[1])  # y1
                    rect.width = int(xyxy[2] - xyxy[0])  # x2 - x1
                    rect.height = int(xyxy[3] - xyxy[1])  # y2 - y1
                    prompt.boxes.append(rect)
                prompt_response = rospy.ServiceProxy("/sam_node/process_prompt", SamPrompt)
                res = prompt_response(prompt)
                seg_msg, vis_img_msg = res.segmentation, res.segmentation_image
                self.segmentation = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding="32SC1")
                self.visualization = self.bridge.imgmsg_to_cv2(vis_img_msg, desired_encoding="rgb8")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
        self.visualization = box_annotator.annotate(scene=self.visualization, detections=detections)
        self.visualization = label_annotator.annotate(
            scene=self.visualization, detections=detections, labels=labels_with_scores
        )
        self.publish_result(
            detections.xyxy, labels, scores, self.segmentation, self.visualization, img_msg.header.frame_id
        )


if __name__ == "__main__":
    rospy.init_node("grounding_dino_node")
    node = GroundingDinoNode()
    rospy.spin()
