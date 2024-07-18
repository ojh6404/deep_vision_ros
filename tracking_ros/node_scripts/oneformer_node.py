#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import supervision as sv
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import Label, LabelArray
from jsk_recognition_msgs.msg import ClassificationResult

from model_config import OneFormerConfig


class OneFormerNode(ConnectionBasedTransport):
    def __init__(self):
        super(OneFormerNode, self).__init__()
        self.task = rospy.get_param("~task", "panoptic")
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.7)
        self.initialize()

        self.bridge = CvBridge()
        self.pub_vis_img = self.advertise("~output/debug_image", Image, queue_size=1)
        self.pub_rects = self.advertise("~output/rects", RectArray, queue_size=1)
        self.pub_labels = self.advertise("~output/labels", LabelArray, queue_size=1)
        self.pub_class = self.advertise("~output/class", ClassificationResult, queue_size=1)
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

    def initialize(self):
        self.detect_flag = False
        self.config = OneFormerConfig.from_rosparam()
        self.predictor = self.config.get_predictor(
            task=self.task,
            confidence_threshold=self.confidence_threshold,
        )
        if self.task == "panoptic":
            self.classes = self.predictor.metadata.stuff_classes
        elif self.task == "instance":
            self.classes = self.predictor.metadata.thing_classes
        elif self.task == "semantic":
            self.classes = self.predictor.metadata.stuff_classes
        else:
            raise ValueError(f"Unknown task: {self.task}")
        self.detect_flag = True

    def publish_result(self, boxes, label_names, scores, mask, vis, frame_id):
        if label_names is not None:
            label_array = LabelArray()
            label_array.labels = [Label(id=i + 1, name=name) for i, name in enumerate(label_names)]
            label_array.header.stamp = rospy.Time.now()
            label_array.header.frame_id = frame_id
            self.pub_labels.publish(label_array)

            class_result = ClassificationResult(
                header=label_array.header,
                classifier=self.config.model_name,
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
            vis_img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
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
            segmentation = None
            xyxys = None
            labels = None
            scores = None
            scores = None

            predictions, visualized_output = self.predictor.run_on_image(self.image, task=self.task)

            if "panoptic_seg" in predictions and self.task == "panoptic":
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
                    scores = self.confidence_threshold * np.ones(len(classes))  # TODO get confidence from model
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

            self.publish_result(
                xyxys,
                labels,
                scores,
                segmentation,
                visualization,
                img_msg.header.frame_id,
            )


if __name__ == "__main__":
    rospy.init_node("oneformer_node")
    node = OneFormerNode()
    rospy.spin()
