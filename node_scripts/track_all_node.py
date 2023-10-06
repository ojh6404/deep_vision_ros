#!/usr/bin/env python
import numpy as np
import torch

import rospy
from cv_bridge import CvBridge
import cv2
import time

from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport


from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from tracking_ros.tracker.base_tracker import BaseTracker
from tracking_ros.utils.util import (
    download_checkpoint,
)
from tracking_ros.utils.painter import mask_painter, point_drawer, bbox_drawer

class TrackAllNode(ConnectionBasedTransport):
    def __init__(self):
        super(TrackAllNode, self).__init__()

        model_dir = rospy.get_param("~model_dir")
        tracker_config_file = rospy.get_param("~tracker_config")
        model_type = rospy.get_param("~model_type", "vit_b")

        sam_checkpoint = download_checkpoint("sam_"+ model_type, model_dir)
        xmem_checkpoint = download_checkpoint("xmem", model_dir)
        self.device = rospy.get_param("~device", "cuda:0")

        # sam
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        # xmem
        self.xmem = BaseTracker(
            xmem_checkpoint, tracker_config_file, device=self.device
        )

        self.bridge = CvBridge()

        self.pub_vis_img = self.advertise("~output_image", Image, queue_size=1)
        self.pub_segmentation_img = self.advertise(
            "~segmentation_mask", Image, queue_size=1
        )

        self.multimask = False

        self.logits = []
        self.masks = []
        self.num_mask = 0

        # for place holder init
        self.image = None
        self.painted_image = None
        self.bbox = None
        self.mask = None
        self.logit = None
        self.template_mask = None


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

    def compose_mask(self, masks):
        """
        input: list of numpy ndarray of 0 and 1, [H, W]
        output: numpy ndarray of 0, 1, ..., len(inputs) [H, W], 0 is background
        """
        template_mask = np.zeros_like(masks[0]).astype(np.uint8)
        for i, mask in enumerate(masks):
            template_mask = np.clip(
                template_mask + mask * (i + 1),
                0,
                i + 1,
            )
            # TODO : checking overlapping mask
            # assert len(np.unique(template_mask)) == (i + 2)

        # assert len(np.unique(template_mask)) == (len(self.masks) + 1)
        return template_mask

    def decompose_mask(self, mask):
        """
        input: numpy ndarray of 0, 1, ..., len(inputs) [H, W], 0 is background
        output: list of numpy ndarray of True and False, [H, W]
        """
        masks = []
        for i in range(len(self.masks)):
            masks.append(mask == (i + 1))
        return masks

    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        if self.template_mask is not None:  # track start
            self.mask, self.logit = self.xmem.track(self.image)
            masks = self.decompose_mask(self.mask)
            seg_mask = self.bridge.cv2_to_imgmsg(
                self.mask.astype(np.uint8), encoding="mono8"
            )
            seg_mask.header.stamp = rospy.Time.now()
            seg_mask.header.frame_id = img_msg.header.frame_id
            self.pub_segmentation_img.publish(seg_mask)
            self.painted_image = self.image.copy()
            for i, mask in enumerate(masks):
                self.painted_image = mask_painter(self.painted_image, mask, i)
        else:  # init
            masks = self.mask_generator.generate(self.image) # dict of masks
            self.masks = [mask["segmentation"].astype(np.uint8) for mask in masks]
            self.template_mask = self.compose_mask(self.masks)
            self.mask, self.logit = self.xmem.track(
                frame=self.image, first_frame_annotation=self.template_mask
            )

        if self.painted_image is not None:
            vis_img_msg = self.bridge.cv2_to_imgmsg(self.painted_image, encoding="rgb8")
            vis_img_msg.header.stamp = rospy.Time.now()
            vis_img_msg.header.frame_id = img_msg.header.frame_id
            self.pub_vis_img.publish(vis_img_msg)


if __name__ == "__main__":
    rospy.init_node("track_all_node")
    node = TrackAllNode()
    rospy.spin()
