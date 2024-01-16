#!/usr/bin/env python

import time
import unittest

import rospy
import rostest
from sensor_msgs.msg import Image


class TestNode(unittest.TestCase):
    def setUp(self):
        pass

    def test_mytest2(self):
        cb_data = {"msg1": None, "msg2": None, "msg3": None, "msg4": None, "msg5": None}

        def all_subscribed() -> bool:
            rospy.logwarn("received: {}".format([not not v for v in cb_data.values()]))
            bool_list = [v is not None for v in cb_data.values()]
            return all(bool_list)

        def cb_debug_image(msg):
            cb_data["msg1"] = msg

        def cb_debug_segimage(msg):
            cb_data["msg2"] = msg

        def cb_info(msg):
            cb_data["msg3"] = msg

        def cb_test_image(msg):
            cb_data["msg4"] = msg

        def cb_test_image_filter(msg):
            cb_data["msg5"] = msg

        rospy.Subscriber("/docker/detic_segmentor/debug_image", Image, cb_debug_image)
        rospy.Subscriber("/docker/detic_segmentor/debug_segmentation_image", Image, cb_debug_segimage)
        rospy.Subscriber("/test_out_image", Image, cb_test_image)
        rospy.Subscriber("/test_out_image_filter", Image, cb_test_image_filter)

        time_out = 40
        for _ in range(time_out):
            time.sleep(1.0)
            if all_subscribed():
                return
        assert False


if __name__ == "__main__":
    rospy.init_node("test_sample")
    rostest.rosrun("detic_ros", "test_node", TestNode)
