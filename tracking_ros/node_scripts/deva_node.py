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

from tracking_ros.cfg import GroundingDINOConfig as ServerConfig
from tracking_ros.model_config import SAMConfig, GroundingDINOConfig, DEVAConfig
from tracking_ros.model_wrapper import GroundingDINOModel, SAMModel, DEVAModel


class DevaNode(ConnectionBasedTransport):
    def __init__(self):
        super(DevaNode, self).__init__()
        self.sam_config = SAMConfig.from_rosparam()
        self.gd_config = GroundingDINOConfig.from_rosparam()
        self.gd_model = GroundingDINOModel(self.gd_config)
        self.deva_config = DEVAConfig.from_rosparam()
        self.deva_model = DEVAModel(self.deva_config)
        self.reconfigure_server = Server(ServerConfig, self.config_cb)
        self.with_bbox = rospy.get_param("~with_bbox", True)

        self.bridge = CvBridge()
        self.pub_segmentation_img = self.advertise("~output/segmentation", Image, queue_size=1)
        self.pub_vis_img = self.advertise("~output/debug_image", Image, queue_size=1)
        self.pub_rects = self.advertise("~output/rects", RectArray, queue_size=1)
        # self.pub_labels = self.advertise("~output/labels", LabelArray, queue_size=1)
        # self.pub_class = self.advertise("~output/class", ClassificationResult, queue_size=1)

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
        self.sam_config = SAMConfig.from_rosparam()
        self.sam_model = SAMModel(self.sam_config)
        self.sam_model.set_model()
        self.classes = [_class.strip() for _class in config.classes.split(";")]
        self.gd_model.set_model(
            self.classes, config.box_threshold, config.text_threshold, config.nms_threshold
        )
        self.deva_model.set_model()
        rospy.loginfo(f"Detecting Classes: {self.classes}")
        self.track_flag = True
        return config

    def publish_result(self, mask, vis, frame_id):
        if mask is not None:
            seg_msg = self.bridge.cv2_to_imgmsg(mask, encoding="32SC1")
            seg_msg.header.stamp = rospy.Time.now()
            seg_msg.header.frame_id = frame_id
            self.pub_segmentation_img.publish(seg_msg)
        if vis is not None:
            vis_img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="rgb8")
            vis_img_msg.header.stamp = rospy.Time.now()
            vis_img_msg.header.frame_id = frame_id
            self.pub_vis_img.publish(vis_img_msg)

    def callback(self, img_msg):
        if self.track_flag:
            image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
            detections, visualization, segmentation = self.deva_model.predict(
                image, self.sam_model, self.gd_model
            )
            self.publish_result(segmentation, visualization, img_msg.header.frame_id)


if __name__ == "__main__":
    rospy.init_node("deva_node")
    node = DevaNode()
    rospy.spin()
