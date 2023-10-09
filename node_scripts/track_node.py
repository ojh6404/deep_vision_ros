#!/usr/bin/env python
import numpy as np
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport

from tracking_ros.tracker.base_tracker import BaseTracker
from tracking_ros.utils.util import download_checkpoint
from tracking_ros.utils.painter import mask_painter

class TrackNode(ConnectionBasedTransport):
    def __init__(self):
        super(TrackNode, self).__init__()

        model_dir = rospy.get_param("~model_dir")
        tracker_config_file = rospy.get_param("~tracker_config")

        xmem_checkpoint = download_checkpoint("xmem", model_dir)
        self.device = rospy.get_param("~device", "cuda:0")
        self.from_detic = rospy.get_param("~mode", None) == "detic"

        # xmem
        self.xmem = BaseTracker(
            xmem_checkpoint, tracker_config_file, device=self.device
        )

        self.bridge = CvBridge()
        self.pub_vis_img = self.advertise("~output_image", Image, queue_size=1)
        self.pub_segmentation_img = self.advertise(
            "~segmentation", Image, queue_size=1
        )

        self.mask = None
        if self.from_detic: # TODO: make this more general
            from detic_ros.msg import SegmentationInfo
            import rosnode
            detic_seg_msg = rospy.wait_for_message("/docker/detic_segmentor/segmentation_info", SegmentationInfo)
            self.classes = detic_seg_msg.detected_classes
            rospy.loginfo("classes: {}".format(self.classes))
            self.template_mask = self.bridge.imgmsg_to_cv2(detic_seg_msg.segmentation, desired_encoding="32SC1")
            # kill detic node for memory
            rosnode.kill_nodes(["/docker/detic_segmentor"])
        else:
            input_seg_msg = rospy.wait_for_message("~input_segmentation", Image)
            self.template_mask = self.bridge.imgmsg_to_cv2(input_seg_msg, desired_encoding="32SC1")
        input_img_msg = rospy.wait_for_message("~input_image", Image)
        self.image = self.bridge.imgmsg_to_cv2(input_img_msg, desired_encoding="rgb8")
        self.num_mask = len(np.unique(self.template_mask)) - 1
        self.mask, self.logit = self.xmem.track(
            frame=self.image, first_frame_annotation=self.template_mask
        )


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
        self.xmem.clear_memory()

    def decompose_mask(self, mask):
        """
        input: numpy ndarray of 0, 1, ..., len(inputs) [H, W], 0 is background
        output: list of numpy ndarray of True and False, [H, W]
        """
        masks = []
        for i in range(self.num_mask):
            masks.append(mask == (i + 1))
        return masks

    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        if self.mask is not None:
            self.mask, self.logit = self.xmem.track(self.image)
            masks = self.decompose_mask(self.mask)
            seg_msg = self.bridge.cv2_to_imgmsg(
                self.mask.astype(np.int32), encoding="32SC1"
            )
            seg_msg.header.stamp = rospy.Time.now()
            seg_msg.header.frame_id = img_msg.header.frame_id
            self.pub_segmentation_img.publish(seg_msg)
            self.painted_image = self.image.copy()
            for i, mask in enumerate(masks):
                self.painted_image = mask_painter(self.painted_image, mask, i)

            vis_img_msg = self.bridge.cv2_to_imgmsg(self.painted_image, encoding="rgb8")
            vis_img_msg.header.stamp = rospy.Time.now()
            vis_img_msg.header.frame_id = img_msg.header.frame_id
            self.pub_vis_img.publish(vis_img_msg)

if __name__ == "__main__":
    rospy.init_node("track_node")
    node = TrackNode()
    rospy.spin()
