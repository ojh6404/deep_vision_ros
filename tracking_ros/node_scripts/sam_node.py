#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PolygonStamped
from std_srvs.srv import Empty, EmptyResponse
from jsk_topic_tools import ConnectionBasedTransport

from model_config import SAMConfig
from utils import draw_prompt, overlay_davis


class SAMNode(ConnectionBasedTransport):
    def __init__(self):
        super(SAMNode, self).__init__()
        self.sam_config = SAMConfig.from_rosparam()
        self.predictor = self.sam_config.get_predictor()

        if self.sam_config.mode == "prompt":  # prompt mode
            self.toggle_prompt_label_service = rospy.Service(
                "/sam_node/sam_controller/toggle_label",
                Empty,
                self.toggle_prompt_label_callback,
            )
            self.clear_prompt_service = rospy.Service(
                "/sam_node/sam_controller/clear_prompt",
                Empty,
                self.clear_prompt_callback,
            )
            self.reset_service = rospy.Service(
                "/sam_node/sam_controller/reset",
                Empty,
                self.reset_callback,
            )
            self.add_mask_service = rospy.Service(
                "/sam_node/sam_controller/add_mask", Empty, self.add_mask_callback
            )
            self.reset_embed_service = rospy.Service(
                "/sam_node/sam_controller/reset_embed",
                Empty,
                self.reset_embed_callback,
            )

            self.publish_mask_service = rospy.Service(
                "/sam_node/sam_controller/publish_mask",
                Empty,
                self.publish_mask_callback,
            )
        else:  # automatic mode
            self.num_slots = rospy.get_param(
                "~num_slots", -1
            )  # number of masks to generate automatically, in order of score

        self.bridge = CvBridge()
        self.pub_segmentation_img = self.advertise("~output/segmentation", Image, queue_size=1)
        self.pub_vis_img = self.advertise("~output/segmentation_image", Image, queue_size=1)

        # initilize prompt
        self.publish_mask = False  # trigger to publish mask
        self.label_mode = True  # True: Positive, False: Negative
        self.prompt_points = []  # points to input
        self.prompt_labels = []  # 1: Positive, 0: Negative
        self.prompt_bbox = None  # bbox to input
        self.prompt_mask = None  # mask when prompting
        self.masks = []  # list of masks stacked from sam predictor [N, H, W]
        self.mask = None  # mask to publish [H, W]
        self.num_mask = 0
        self.multimask = False
        self.embedded_image = None

    def subscribe(self):
        self.sub_image = rospy.Subscriber(
            "~input_image",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.subs = [self.sub_image]
        if self.sam_config.mode == "prompt":
            self.sub_point = rospy.Subscriber(
                "~input_point",
                PointStamped,
                self.point_callback,
                queue_size=1,
                buff_size=2**24,
            )
            self.sub_bbox = rospy.Subscriber(
                "~input_bbox",
                PolygonStamped,
                self.bbox_callback,
                queue_size=1,
                buff_size=2**24,
            )
            self.subs += [self.sub_point, self.sub_bbox]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def clear_prompt(self):
        self.prompt_points.clear()
        self.prompt_labels.clear()
        self.prompt_bbox = None
        self.prompt_mask = None
        self.logit = None

    def clear_prompt_callback(self, srv):
        rospy.loginfo("Clear prompt input")
        self.clear_prompt()
        return EmptyResponse()

    def reset_callback(self, srv):
        rospy.loginfo("Reset prompt input and masks")
        self.masks.clear()
        self.clear_prompt()
        self.visualization = None
        self.mask = None
        self.num_mask = 0
        return EmptyResponse()

    def add_mask_callback(self, srv):
        if self.prompt_mask is None:
            rospy.logwarn("No mask to add")
            self.clear_prompt()
            return EmptyResponse()
        self.masks.append(self.prompt_mask)
        self.num_mask += 1
        if self.mask is None:
            self.mask = self.prompt_mask * (self.num_mask)
        else:
            self.mask[self.prompt_mask] = self.num_mask
        self.clear_prompt()
        rospy.loginfo("Mask added")
        return EmptyResponse()

    def reset_embed_callback(self, srv):
        rospy.loginfo("Embedding image for segmentation")
        if self.embedded_image is not None:
            self.predictor.reset_image()
        self.embedded_image = self.image.copy()
        self.predictor.set_image(self.embedded_image)
        return EmptyResponse()

    def publish_mask_callback(self, srv):  # TODO
        rospy.loginfo("Publish SAM mask")
        self.publish_mask = not self.publish_mask
        return EmptyResponse()

    def toggle_prompt_label_callback(self, srv):
        self.label_mode = not self.label_mode
        rospy.loginfo("Toggle prompt label to {}".format(self.label_mode))
        res = EmptyResponse()
        return res

    def point_callback(self, point_msg):
        # if point x and point y is out of image shape, just pass
        point_x = int(point_msg.point.x)  # x is within 0 ~ width
        point_y = int(point_msg.point.y)  # y is within 0 ~ height

        if point_x < 1 or point_x > (self.image.shape[1] - 1) or point_y < 1 or point_y > (self.image.shape[0] - 1):
            rospy.logwarn("point {} is out of image shape".format([point_x, point_y]))
            return

        point = [point_x, point_y]
        label = 1 if self.label_mode else 0

        rospy.loginfo("point {} and label {} added".format(point, label))
        self.prompt_points.append(point)
        self.prompt_labels.append(label)

        self.prompt_mask, self.logit = self.process_prompt(
            points=np.array(self.prompt_points),
            labels=np.array(self.prompt_labels),
            bbox=np.array(self.prompt_bbox) if self.prompt_bbox is not None else None,
            multimask=self.multimask,
        )

    def bbox_callback(self, bbox_msg):
        # clip bbox
        x1 = bbox_msg.polygon.points[0].x
        y1 = bbox_msg.polygon.points[0].y
        x2 = bbox_msg.polygon.points[1].x
        y2 = bbox_msg.polygon.points[1].y

        # x2 and y2 should be larger than x1 and y1 for xyxy format
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x_l = max(0, x1)
        y_l = max(0, y1)
        x_r = min(self.image.shape[1], x2)
        y_r = min(self.image.shape[0], y2)

        rospy.loginfo("bbox {} {} {} {}".format(x_l, y_l, x_r, y_r))
        self.prompt_bbox = [x_l, y_l, x_r, y_r]

        self.prompt_mask, self.logit = self.process_prompt(
            points=None,
            labels=None,
            bbox=np.array(self.prompt_bbox),
            multimask=self.multimask,
        )

    def process_prompt(
        self,
        points=None,
        bbox=None,
        labels=None,
        mask_input=None,
        multimask: bool = True,
    ):
        if self.embedded_image is None:
            self.embedded_image = self.image
            self.predictor.set_image(self.embedded_image)
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            mask_input=mask_input,  # TODO
            multimask_output=multimask,
        )  # [N, H, W], B : number of prompts, N : number of masks recommended
        mask, logit = (
            masks[np.argmax(scores)],
            logits[np.argmax(scores)],
        )  # choose the best mask [H, W]

        # refine mask using logit
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            mask_input=logit[None, :, :],
            multimask_output=multimask,
        )
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)]
        return mask, logit

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

    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        if self.publish_mask:
            self.publish_result(self.mask, self.visualization, img_msg.header.frame_id)
        else:
            if self.sam_config.mode == "prompt":  # when prompt mode
                self.visualization = self.image.copy()
                if self.mask is not None:  # if mask exists
                    if self.prompt_mask is not None:  # if prompt mask exists
                        paint_mask = self.mask.copy()
                        paint_mask[self.prompt_mask] = self.num_mask + 1
                        self.visualization = overlay_davis(self.visualization, paint_mask)
                    else:  # if prompt mask does not exist
                        self.visualization = overlay_davis(self.visualization, self.mask)
                else:  # when initial state
                    if self.prompt_mask is not None:  # if prompt mask exists
                        self.visualization = overlay_davis(self.visualization, self.prompt_mask.astype(np.uint8))
                self.visualization = draw_prompt(
                    self.visualization,
                    self.prompt_points,
                    self.prompt_labels,
                    self.prompt_bbox,
                    f"Prompt {self.num_mask}",
                )
                self.publish_result(None, self.visualization, img_msg.header.frame_id)

            else:
                masks = self.predictor.generate(self.image)  # dict of masks
                self.masks = [mask["segmentation"] for mask in masks]  # list of [H, W]
                if self.num_slots > 0:
                    self.masks = [mask["segmentation"] for mask in masks][: self.num_slots]
                self.mask = np.zeros_like(self.masks[0]).astype(np.uint8)  # [H, W]
                for i, mask in enumerate(self.masks):
                    self.mask = np.clip(
                        self.mask + mask * (i + 1),
                        0,
                        i + 1,
                    )
                self.publish_mask = True
                self.visualization = overlay_davis(self.image.copy(), self.mask)


if __name__ == "__main__":
    rospy.init_node("sam_node")
    node = SAMNode()
    rospy.spin()
