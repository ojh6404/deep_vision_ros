#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import rospy
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from sensor_msgs.msg import Image

from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import Label, LabelArray

from deep_vision_ros.cfg import GroundingDINOConfig as ServerConfig
from deep_vision_ros.model_config import GroundingDINOConfig, MASAConfig
from deep_vision_ros.model_wrapper import GroundingDINOModel, MASAModel


class MASANode(ConnectionBasedTransport):
    def __init__(self):
        super(MASANode, self).__init__()
        self.fp16 = rospy.get_param("~fp16", True)
        self.gd_config = GroundingDINOConfig.from_rosparam()
        self.gd_model = GroundingDINOModel(self.gd_config)
        self.masa_config = MASAConfig.from_rosparam()
        self.masa_model = MASAModel(self.masa_config)
        self.reconfigure_server = Server(ServerConfig, self.config_cb)
        self.with_bbox = rospy.get_param("~with_bbox", True)

        self.bridge = CvBridge()
        self.pub_segmentation_img = self.advertise("~output/segmentation", Image, queue_size=1)
        self.pub_vis_img = self.advertise("~output/debug_image", Image, queue_size=1)
        self.pub_rects = self.advertise("~output/rects", RectArray, queue_size=1)
        self.pub_labels = self.advertise("~output/labels", LabelArray, queue_size=1)
        self.pub_class = self.advertise("~output/class", ClassificationResult, queue_size=1)

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
        self.track_flag = False
        self.classes = [_class.strip() for _class in config.classes.split(";")]
        self.gd_model.set_model(
            self.classes, config.box_threshold, config.text_threshold, config.nms_threshold
        )
        self.masa_model.set_model(classes=self.classes, fp16=self.fp16)
        rospy.loginfo(f"Detecting Classes: {self.classes}")
        self.track_flag = True
        return config

    def publish_result(
        self,
        boxes=None,
        label_names=None,
        scores=None,
        mask=None,
        vis=None,
        frame_id="camera_rgb_optical_frame",
    ):
        if label_names is not None:
            label_array = LabelArray()
            label_array.labels = [Label(id=i + 1, name=name) for i, name in enumerate(label_names)]
            label_array.header.stamp = rospy.Time.now()
            label_array.header.frame_id = frame_id
            self.pub_labels.publish(label_array)

            # class_result = ClassificationResult(
            #     header=label_array.header,
            #     classifier=self.gd_config.model_name,
            #     target_names=self.classes,
            #     labels=[self.classes.index(name.split("_")[0]) for name in label_names],
            #     label_names=label_names,
            #     label_proba=scores,
            # )
            # self.pub_class.publish(class_result)

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
        if self.track_flag:
            image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            detections, visualization = self.masa_model.predict(image, self.gd_model)

            self.publish_result(
                boxes=detections.xyxy,
                label_names=[
                    f"{self.gd_model.classes[label_id]}_{instance_id}"
                    for label_id, instance_id in zip(detections.class_id, detections.tracker_id)
                ],
                vis=visualization,
                frame_id=img_msg.header.frame_id,
            )


if __name__ == "__main__":
    rospy.init_node("masa_node")
    node = MASANode()
    rospy.spin()
