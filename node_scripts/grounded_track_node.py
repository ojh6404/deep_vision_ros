#!/usr/bin/env python
import numpy as np
import torch

import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport


from segment_anything import sam_model_registry, SamPredictor
from tracking_ros.tracker.base_tracker import BaseTracker
from tracking_ros.utils.util import (
    download_checkpoint,
)
from tracking_ros.utils.painter import mask_painter, point_drawer, bbox_drawer
from tracking_ros.utils.dino_utils import get_grounded_bbox

from groundingdino.util.inference import load_model

class GroundedTrackNode(ConnectionBasedTransport):
    def __init__(self):
        super(GroundedTrackNode, self).__init__()

        model_dir = rospy.get_param("~model_dir")
        tracker_config_file = rospy.get_param("~tracker_config")
        model_type = rospy.get_param("~model_type", "vit_b")

        sam_checkpoint = download_checkpoint("sam_"+ model_type, model_dir)
        xmem_checkpoint = download_checkpoint("xmem", model_dir)
        dino_checkpoint = download_checkpoint("dino", model_dir)
        self.device = rospy.get_param("~device", "cuda:0")

        # grounding dino
        dino_config = rospy.get_param("~dino_config")
        self.text_prompt = rospy.get_param("~text_prompt")
        self.grounding_dino = load_model(dino_config, dino_checkpoint, device=self.device)
        self.box_threshold = rospy.get_param("~box_threshold", 0.35)
        self.text_threshold = rospy.get_param("~text_threshold", 0.25)

        # sam
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        # xmem
        self.xmem = BaseTracker(
            xmem_checkpoint, tracker_config_file, device=self.device
        )

        self.bridge = CvBridge()

        self.pub_vis_img = self.advertise("~output_image", Image, queue_size=1)
        self.pub_segmentation_img = self.advertise(
            "~segmentation_mask", Image, queue_size=1
        )
        self.label_mode = True # True: Positive, False: Negative

        self.points = []
        self.labels = []
        self.multimask = False

        self.logits = []
        self.masks = []
        self.num_mask = 0

        # for place holder init
        self.embedded_image = None
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





    def process_prompt(
        self,
        points=None,
        bbox=None,
        labels=None,
        mask_input=None,
        multimask:bool=True,
    ):
        """
        it is used in first frame
        return: mask, logit, painted image(mask+point)
        """
        bbox = torch.Tensor(bbox).to(self.device)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            bbox, self.image.shape[:2])

        
        prompts = dict(
            point_coords= None,
            point_labels= None,
            boxes= transformed_boxes,
            # mask_input= mask_input, # TODO
            multimask_output= multimask,
        )
        masks, scores, logits = self.predictor.predict_torch(
            **prompts) # [N, H, W], B : number of prompts, N : number of masks recommended

        return masks, logits


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
            assert len(np.unique(template_mask)) == (i + 2)

        assert len(np.unique(template_mask)) == (len(self.masks) + 1)
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

            bboxes = get_grounded_bbox(
                model = self.grounding_dino,
                image=self.image,
                text_prompt=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

            self.predictor.set_image(self.image)
            self.masks, self.logits = self.process_prompt(
                None,
                bboxes,
                None,
                None,
                False)

            self.masks = [mask.squeeze(0).cpu().numpy().astype(np.uint8) for mask in self.masks]
            self.template_mask = self.compose_mask(self.masks)

            masks = self.decompose_mask(self.template_mask)

            painted_image = self.image.copy()
            for i, mask in enumerate(masks):
                mask = mask.astype(np.bool8)
                painted_image = mask_painter(painted_image, mask, i)


            self.mask, self.logit = self.xmem.track(
                frame=self.image, first_frame_annotation=self.template_mask
            )

            # visualize
            self.painted_image = self.image.copy()
            for i, mask in enumerate(self.masks + [self.mask]):
                self.painted_image = mask_painter(self.painted_image, mask, i)
            self.painted_image = point_drawer(
                self.painted_image,
                self.points,
                self.labels
            )
            self.painted_image = bbox_drawer(self.painted_image, self.bbox)


        vis_img_msg = self.bridge.cv2_to_imgmsg(self.painted_image, encoding="rgb8")
        vis_img_msg.header.stamp = rospy.Time.now()
        vis_img_msg.header.frame_id = img_msg.header.frame_id

        self.pub_vis_img.publish(vis_img_msg)




if __name__ == "__main__":
    rospy.init_node("grounded_track_node")
    node = GroundedTrackNode()
    rospy.spin()
