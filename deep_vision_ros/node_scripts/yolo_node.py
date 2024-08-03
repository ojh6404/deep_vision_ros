#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import Label, LabelArray
from jsk_recognition_msgs.msg import ClassificationResult

from vision_anything.config.model_config import YOLOConfig, SAMConfig
from vision_anything.model.model_wrapper import YOLOModel, SAMModel
from vision_anything.utils.vis_utils import overlay_davis, BOX_ANNOTATOR, LABEL_ANNOTATOR
from deep_vision_ros.cfg import YOLOConfig as ServerConfig


class YOLONode(ConnectionBasedTransport):
    def __init__(self):
        super(YOLONode, self).__init__()
        self.get_mask = rospy.get_param("~get_mask", False)
        self.yolo_config = YOLOConfig.from_args(
            model_type=rospy.get_param("~yolo_model_type", "yolov8x_worldv2"),
            device=rospy.get_param("~device", "cuda:0"),
        )
        self.model = YOLOModel(self.yolo_config)
        if self.get_mask:
            self.sam_config = SAMConfig.from_args(
                model_type=rospy.get_param("~sam_model_type", "vit_t"),
                mode="prompt",
                device=rospy.get_param("~device", "cuda:0"),
            )
            self.sam_model = SAMModel(self.sam_config)
        self.reconfigure_server = Server(ServerConfig, self.config_cb)

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
        self.model.set_model(self.classes, config.box_threshold, config.nms_threshold)
        if self.get_mask:
            self.sam_model.set_model()
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
                classifier=self.yolo_config.model_name,
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
            image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            detections, visualization = self.model.predict(image)
            labels = [self.classes[i] for i in detections.class_id]
            segmentation = None

            if self.get_mask:
                segmentation, visualization = self.sam_model.predict(image=image, boxes=detections.xyxy)
                visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
                visualization = LABEL_ANNOTATOR.annotate(
                    scene=visualization, detections=detections, labels=labels
                )
            self.publish_result(
                detections.xyxy,
                labels,
                detections.confidence,
                segmentation,
                visualization,
                img_msg.header.frame_id,
            )


if __name__ == "__main__":
    rospy.init_node("yolo_node")
    node = YOLONode()
    rospy.spin()
