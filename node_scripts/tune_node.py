#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
import os

from sensor_msgs.msg import Image
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

class ImageTuneNode(object):
    def __init__(self):
        super(ImageTuneNode, self).__init__()
        self.config = rospy.get_param("~config", None)
        if self.config:
            # check if config file exists
            if os.path.exists(self.config):
                self.image_tuner = HSVBlurCropResolFilter.from_yaml(self.config)
            else:
                rospy.loginfo("Config file not found, creating new one")
                self.image_tuner = None
        else:
            self.image_tuner = None
        self.bridge = CvBridge()
        self.sub_img = rospy.Subscriber("~in", Image, self.callback, queue_size=1)
        self.pub_tuned_img = rospy.Publisher("~out", Image, queue_size=1)

    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        if self.image_tuner is not None:
            self.repub_image = self.image_tuner(self.image)
            tuned_img_msg = self.bridge.cv2_to_imgmsg(self.repub_image, encoding="rgb8")
            tuned_img_msg.header.stamp = rospy.Time.now()
            tuned_img_msg.header.frame_id = img_msg.header.frame_id
            self.pub_tuned_img.publish(tuned_img_msg)
        else:
            tunable = HSVBlurCropResolFilter.from_image(self.image)
            rospy.loginfo("Press q to finish tuning")
            tunable.launch_window()
            tunable.start_tuning(self.image)
            tunable.dump_yaml(self.config)
            self.image_tuner = HSVBlurCropResolFilter.from_yaml(self.config)

if __name__ == "__main__":
    rospy.init_node("image_tune_node")
    node = ImageTuneNode()
    rospy.spin()
