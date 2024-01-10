#!/usr/bin/env python

import numpy as np
import cv2
import rospy
import rosnode
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PolygonStamped
from std_srvs.srv import Empty, EmptyResponse
from jsk_topic_tools import ConnectionBasedTransport

from gui.interactive_utils import overlay_davis

def point_drawer(image, points, labels, radius=5):
    if points == []:
        return image
    for i, point in enumerate(points):
        if labels[i] == 1:
            color = [0, 0, 255]
        else:
            color = [255, 0, 0]
        image = cv2.circle(image, tuple(point), radius, color, -1)
    return image.astype(np.uint8)

def bbox_drawer(image, bbox, color=None):
    if bbox is None:
        return image
    if color is None:
        color = [0, 255, 0]
    image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]) ), color, 2)
    return image.astype(np.uint8)

class SAMNode(ConnectionBasedTransport):
    def __init__(self):
        super(SAMNode, self).__init__()
        self.device = rospy.get_param("~device", "cuda:0")
        self.prompt_mode = rospy.get_param("~mode", "interactive") == "interactive"
        model_type = rospy.get_param("~model_type", "vit_h_hq")
        sam_checkpoint = rospy.get_param("~model_path")
        is_hq = "hq" in model_type

        # sam
        if is_hq:
            from segment_anything_hq import (
                sam_model_registry,
                SamPredictor,
                SamAutomaticMaskGenerator,
            )
        else:
            from segment_anything import (
                sam_model_registry,
                SamPredictor,
                SamAutomaticMaskGenerator,
            )

        self.sam = sam_model_registry[model_type[:5]](checkpoint=sam_checkpoint)
        self.sam.to(self.device)

        # prompt mode
        if self.prompt_mode:
            self.predictor = SamPredictor(self.sam)
            self.toggle_prompt_label_service = rospy.Service(
                "/tracking_ros/toggle_label", Empty, self.toggle_prompt_label_callback
            )
            self.clear_points_service = rospy.Service(
                "/tracking_ros/clear_points", Empty, self.clear_points_callback
            )
            self.clear_masks_service = rospy.Service(
                "/tracking_ros/clear_masks", Empty, self.clear_masks_callback
            )
            self.add_mask_service = rospy.Service(
                "/tracking_ros/add_mask", Empty, self.add_mask_callback
            )
            self.reset_embed_service = rospy.Service(
                "/tracking_ros/reset_embed", Empty, self.reset_embed_callback
            )

            self.track_trigger_service = rospy.Service(
                "/tracking_ros/track_trigger", Empty, self.track_trigger_callback
            )
            self.pub_vis_img = self.advertise("~output_image", Image, queue_size=1)
        else:   # automatic mode
            self.predictor = SamAutomaticMaskGenerator(self.sam)
            self.num_slots = rospy.get_param("~num_slots", -1)

        self.bridge = CvBridge()
        self.pub_segmentation_img = self.advertise("~segmentation", Image, queue_size=1)
        self.label_mode = True  # True: Positive, False: Negative
        self.track_trigger = False

        self.points = []
        self.labels = []
        self.multimask = False

        self.masks = []
        self.num_mask = 0

        # for place holder init
        self.embedded_image = None
        self.prompt_mask = None
        self.prompt_image = None
        self.image = None
        self.bbox = None
        self.template_mask = None

    def subscribe(self):
        self.sub_image = rospy.Subscriber(
            "~input_image",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.subs = [self.sub_image]
        if self.prompt_mode:
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



    def clear_points_callback(self, srv):
        rospy.loginfo("Clear prompts")
        self.points.clear()
        self.labels.clear()
        self.bbox = None
        self.prompt_mask = None
        self.logit = None
        res = EmptyResponse()
        return res

    def clear_masks_callback(self, srv):
        rospy.loginfo("Clear masks")
        self.masks.clear()
        self.prompt_mask = None
        self.logit = None
        self.painted_image = None
        res = EmptyResponse()
        return res

    def add_mask_callback(self, srv):
        if self.prompt_mask is None:
            rospy.logwarn("No mask to add")
            self.points.clear()
            self.labels.clear()
            return res
        self.masks.append(self.prompt_mask)
        self.points.clear()
        self.labels.clear()
        self.bbox = None
        self.num_mask += 1
        if self.template_mask is None:
            self.template_mask = self.prompt_mask * (self.num_mask)
        else:
            self.template_mask[self.prompt_mask] = self.num_mask
        rospy.loginfo("Mask added")
        res = EmptyResponse()
        return res

    def reset_embed_callback(self, srv):
        rospy.loginfo("Embedding image for segmentation")
        if self.embedded_image is not None:
            self.predictor.reset_image()
        self.embedded_image = self.image
        self.predictor.set_image(self.image)
        res = EmptyResponse()
        return res

    def track_trigger_callback(self, srv):  # TODO
        rospy.loginfo("Tracking start...")
        self.track_trigger = True
        res = EmptyResponse()
        return res

    def toggle_prompt_label_callback(self, srv):
        self.label_mode = not self.label_mode
        rospy.loginfo("Toggle prompt label to {}".format(self.label_mode))
        res = EmptyResponse()
        return res

    def point_callback(self, point_msg):
        # if point x and point y is out of image shape, just pass
        point_x = int(point_msg.point.x)  # x is within 0 ~ width
        point_y = int(point_msg.point.y)  # y is within 0 ~ height

        if (
            point_x < 1
            or point_x > (self.image.shape[1] - 1)
            or point_y < 1
            or point_y > (self.image.shape[0] - 1)
        ):
            rospy.logwarn("point {} is out of image shape".format([point_x, point_y]))
            return

        point = [point_x, point_y]
        label = 1 if self.label_mode else 0

        rospy.loginfo("point {} and label {} added".format(point, label))
        self.points.append(point)
        self.labels.append(label)

        if self.embedded_image is None:
            self.predictor.set_image(self.image)
            self.embedded_image = self.image

        self.prompt_mask, self.logit = self.process_prompt(
            points=np.array(self.points),
            labels=np.array(self.labels),
            bbox=np.array(self.bbox) if self.bbox is not None else None,
            multimask=self.multimask,
        )

    def bbox_callback(self, bbox_msg):
        x_l = bbox_msg.polygon.points[0].x
        y_l = bbox_msg.polygon.points[0].y
        x_r = bbox_msg.polygon.points[1].x
        y_r = bbox_msg.polygon.points[1].y

        # clip bbox
        x_l = max(0, x_l)
        y_l = max(0, y_l)
        x_r = min(self.image.shape[1], x_r)
        y_r = min(self.image.shape[0], y_r)

        rospy.loginfo("bbox {} {} {} {}".format(x_l, y_l, x_r, y_r))
        self.bbox = [x_l, y_l, x_r, y_r]

        if self.embedded_image is None:
            self.predictor.set_image(self.image)
            self.embedded_image = self.image

        self.prompt_mask, self.logit = self.process_prompt(
            points=None,
            labels=None,
            bbox=np.array(self.bbox) if self.bbox is not None else None,
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
        """
        it is used in first frame
        return: mask, logit, painted image(mask+point)
        """
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
        return template_mask

    def kill_nodes(self):
        # kill node
        nodes_list = rosnode.get_node_names()
        for node in nodes_list:
            if "image_view" in node or "prompter" in node:
                rospy.loginfo("kill {}".format(node))
                rosnode.kill_nodes([node])
        rospy.signal_shutdown("Tracking done")

    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        if self.track_trigger:
            # TODO free memory
            seg_msg = self.bridge.cv2_to_imgmsg(
                self.template_mask.astype(np.int32), encoding="32SC1"
            )
            self.pub_segmentation_img.publish(seg_msg)

            # self.kill_nodes()
        else:  # init
            if self.prompt_mode:
                self.painted_image = self.image.copy()
                if self.template_mask is not None:
                    paint_mask = self.template_mask.copy()
                    paint_mask[self.prompt_mask] = self.num_mask + 1
                    self.painted_image = overlay_davis(self.painted_image, paint_mask)
                else:
                    if self.prompt_mask is not None:
                        self.painted_image = overlay_davis(
                            self.painted_image, self.prompt_mask.astype(np.uint8)
                        )

                self.painted_image = point_drawer(
                    self.painted_image, self.points, self.labels
                )
                self.painted_image = bbox_drawer(self.painted_image, self.bbox)

                vis_img_msg = self.bridge.cv2_to_imgmsg(
                    self.painted_image, encoding="rgb8"
                )
                vis_img_msg.header.stamp = rospy.Time.now()
                vis_img_msg.header.frame_id = img_msg.header.frame_id
                self.pub_vis_img.publish(vis_img_msg)
            else:
                masks = self.predictor.generate(self.image)  # dict of masks
                self.masks = [mask["segmentation"] for mask in masks]
                if self.num_slots > 0:
                    self.masks = [mask["segmentation"] for mask in masks][
                        : self.num_slots
                    ]
                self.template_mask = self.compose_mask(self.masks)
                self.track_trigger = True


if __name__ == "__main__":
    rospy.init_node("sam_node")
    node = SAMNode()
    rospy.spin()
