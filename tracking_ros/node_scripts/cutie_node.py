#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import supervision as sv
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse
from tracking_ros_utils.srv import CutiePrompt, CutiePromptResponse

from tracking_ros.model_config import CutieConfig
from tracking_ros.utils import overlay_davis

BOX_ANNOTATOR = sv.BoxAnnotator()


class CutieNode(object):  # should not be ConnectionBasedNode cause Cutie tracker needs continuous input
    def __init__(self):
        super(CutieNode, self).__init__()
        self.with_bbox = rospy.get_param("~with_bbox", False)
        self.bridge = CvBridge()
        image, mask = self.get_oneshot_prompt()
        self.track_flag = self.initialize(image, mask)

        self.sub_image = rospy.Subscriber(
            "~input_image",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.pub_vis_img = rospy.Publisher("~output/segmentation_image", Image, queue_size=1)
        self.pub_segmentation_img = rospy.Publisher("~output/segmentation", Image, queue_size=1)

        # reset tracking service
        self.reset_service = rospy.Service("~reset", Empty, self.reset_callback)

        self.process_prompt_service = rospy.Service(
            "~process_prompt",
            CutiePrompt,
            self.prompt_service_callback,
        )

    def get_oneshot_prompt(self):
        try:
            # oneshot subscribe initial image and segmentation
            input_seg_msg = rospy.wait_for_message("~input_segmentation", Image, timeout=5)
            input_img_msg = rospy.wait_for_message("~input_image", Image, timeout=5)
            mask = self.bridge.imgmsg_to_cv2(input_seg_msg, desired_encoding="32SC1")
            image = self.bridge.imgmsg_to_cv2(input_img_msg, desired_encoding="rgb8")
            return image, mask
        except rospy.ROSException:
            rospy.logwarn("No message received in 5 seconds")
            return None, None

    def prompt_service_callback(self, req):
        rospy.loginfo("Processing prompt and resetting Cutie tracker")
        self.track_flag = False
        input_seg_msg = req.segmentation
        input_img_msg = req.image
        mask = self.bridge.imgmsg_to_cv2(input_seg_msg, desired_encoding="32SC1")
        image = self.bridge.imgmsg_to_cv2(input_img_msg, desired_encoding="rgb8")
        self.track_flag = self.initialize(image, mask)
        return CutiePromptResponse(result=True)

    def reset_callback(self, req):
        rospy.loginfo("Resetting Cutie tracker")
        self.track_flag = False
        image, mask = self.get_oneshot_prompt()
        self.initialize(image, mask)
        self.track_flag = self.initialize(image, mask)
        return EmptyResponse()

    @torch.inference_mode()
    def initialize(self, image, mask):
        if image is None or mask is None:
            return False

        self.cutie_config = CutieConfig.from_rosparam()
        self.predictor = self.cutie_config.get_predictor()

        # initialize the model with the mask
        with torch.cuda.amp.autocast(enabled=True):
            image_torch = (
                torch.from_numpy(image.transpose(2, 0, 1)).float().to(self.cutie_config.device, non_blocking=True) / 255
            )
            # initialize with the mask
            mask_torch = (
                F.one_hot(
                    torch.from_numpy(mask).long(),
                    num_classes=len(np.unique(mask)),
                )
                .permute(2, 0, 1)
                .float()
                .to(self.cutie_config.device)
            )
            # the background mask is not fed into the model
            self.mask = self.predictor.step(image_torch, mask_torch[1:], idx_mask=False)
        return True

    def publish_result(self, mask, vis, frame_id):
        if mask is not None:
            seg_msg = self.bridge.cv2_to_imgmsg(mask.astype(np.int32), encoding="32SC1")
            seg_msg.header.stamp = rospy.Time.now()
            seg_msg.header.frame_id = frame_id
            self.pub_segmentation_img.publish(seg_msg)
        if vis is not None:
            vis_img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="rgb8")
            vis_img_msg.header.stamp = rospy.Time.now()
            vis_img_msg.header.frame_id = frame_id
            self.pub_vis_img.publish(vis_img_msg)

    @torch.inference_mode()
    def callback(self, img_msg):
        if self.track_flag:
            self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
            with torch.cuda.amp.autocast(enabled=True):
                image_torch = (
                    torch.from_numpy(self.image.transpose(2, 0, 1))
                    .float()
                    .to(self.cutie_config.device, non_blocking=True)
                    / 255
                )
                prediction = self.predictor.step(image_torch)
                self.mask = torch.max(prediction, dim=0).indices.cpu().numpy().astype(np.uint8)
                self.visualization = overlay_davis(self.image.copy(), self.mask)
                if self.with_bbox and len(np.unique(self.mask)) > 1:
                    masks = []
                    for i in range(1, len(np.unique(self.mask))):
                        masks.append((self.mask == i).astype(np.uint8))

                    self.masks = np.stack(masks, axis=0)
                    xyxy = sv.mask_to_xyxy(self.masks)  # [N, 4]
                    detections = sv.Detections(
                        xyxy=xyxy,
                        mask=self.masks,
                        tracker_id=np.arange(len(xyxy)),
                    )
                    self.visualization = BOX_ANNOTATOR.annotate(
                        scene=self.visualization,
                        detections=detections,
                        labels=[f"ObjectID : {i}" for i in range(len(xyxy))],
                    )
            self.publish_result(self.mask, self.visualization, img_msg.header.frame_id)


if __name__ == "__main__":
    rospy.init_node("cutie_node")
    node = CutieNode()
    rospy.spin()
