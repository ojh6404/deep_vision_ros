#!/usr/bin/env python
import numpy as np
import torch

import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport

from segment_anything import sam_model_registry, SamPredictor
from tracking_ros.utils.util import download_checkpoint
from tracking_ros.utils.dino_utils import get_grounded_bbox
from groundingdino.util.inference import load_model

class GroundingDinoNode(ConnectionBasedTransport):
    def __init__(self):
        super(GroundingDinoNode, self).__init__()

        model_dir = rospy.get_param("~model_dir")
        model_type = rospy.get_param("~model_type", "vit_b")

        sam_checkpoint = download_checkpoint("sam_"+ model_type, model_dir)
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

        self.bridge = CvBridge()
        self.pub_segmentation_img = self.advertise(
            "~segmentation", Image, queue_size=1
        )

        # for place holder init
        self.embedded_image = None
        self.image = None
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

    def process_prompt(
        self,
        points=None,
        bbox=None,
        labels=None,
        mask_input=None,
        multimask:bool=True,
    ):
        bbox = torch.Tensor(bbox).to(self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            bbox, self.image.shape[:2])
        prompts = dict(
            point_coords= None,
            point_labels= None,
            boxes= transformed_boxes,
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
            # assert len(np.unique(template_mask)) == (i + 2)

        # assert len(np.unique(template_mask)) == (len(self.masks) + 1)
        return template_mask

    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        if self.template_mask is not None:  # track start
            if self.predictor:
                del self.grounding_dino
                del self.predictor
                del self.sam
            seg_msg = self.bridge.cv2_to_imgmsg(
                self.template_mask.astype(np.int32), encoding="32SC1"
            )
            seg_msg.header.stamp = rospy.Time.now()
            seg_msg.header.frame_id = img_msg.header.frame_id
            self.pub_segmentation_img.publish(seg_msg)

            rospy.signal_shutdown("grounded detection done")
        else:  # init
            bboxes, phrases = get_grounded_bbox(
                model = self.grounding_dino,
                image=self.image,
                text_prompt=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            for i, bbox in enumerate(bboxes):
                rospy.loginfo("bbox: {}, phrase: {}".format(bbox, phrases[i]))
            self.predictor.set_image(self.image)
            self.masks, self.logits = self.process_prompt(
                None,
                bboxes,
                None,
                None,
                False)
            self.masks = [mask.squeeze(0).cpu().numpy().astype(np.uint8) for mask in self.masks]
            self.template_mask = self.compose_mask(self.masks)

if __name__ == "__main__":
    rospy.init_node("grounding_dino_node")
    node = GroundingDinoNode()
    rospy.spin()
