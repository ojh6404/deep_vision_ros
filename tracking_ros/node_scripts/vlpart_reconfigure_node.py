#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rosnode
import subprocess

from dynamic_reconfigure.server import Server
from tracking_ros.cfg import VLPartConfig as ServerConfig


class VLPartReconfigureNode(object):
    def __init__(self):
        super(VLPartReconfigureNode, self).__init__()
        self.node_name = rospy.get_param("~node_name", "/vlpart_node")
        self.input_image = rospy.get_param("~input_image", "/camera/rgb/image_rect_color")
        self.classes = rospy.get_param("~classes", "bottle cap; cup handle;")
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        self.vocabulary = rospy.get_param("~vocabulary", "custom")
        self.device = rospy.get_param("~device", "cuda:0")
        self.use_sam = rospy.get_param("~use_sam", False)
        self.reconfigure_server = Server(ServerConfig, self.config_cb)

    def config_cb(self, config, level):
        self.classes = config.classes
        self.confidence_threshold = config.confidence_threshold
        self.vocabulary = config.vocabulary
        rospy.loginfo(f"Detecting Classes: {self.classes}")

        # rosrun using subprocess cause vlpart is not configurable when it is running
        if (
            self.node_name in rosnode.get_node_names()
        ):  # Check if node exists, if it does, kill the node to reconfigure
            rospy.loginfo("Node exists, killing node")
            rosnode.kill_nodes([self.node_name])
        else:  # If node does not exist, run the node
            rospy.loginfo("Node does not exist")
        subprocess.Popen(
            """rosrun tracking_ros vlpart_node.py \
                __name:={} \
                ~input_image:={} \
                _vocabulary:={} \
                _classes:='{}' \
                _device:={} \
                _use_sam:={} \
            """.format(
                self.node_name, self.input_image, self.vocabulary, self.classes, self.device, self.use_sam
            ),
            shell=True,
        )
        return config


if __name__ == "__main__":
    rospy.init_node("vlpart_reconfigure_node")
    node_manager_node = VLPartReconfigureNode()
    rospy.spin()
