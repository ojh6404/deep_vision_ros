#!/usr/bin/env python

import numpy as np
import torch

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from omegaconf import open_dict
from hydra import compose, initialize

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.args_utils import get_dataset_cfg

from gui.interactive_utils import overlay_davis, image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch


class CutieNode(object): # should not be ConnectionBasedNode cause Cutie tracker needs continuous input
    def __init__(self):
        super(CutieNode, self).__init__()
        self.device = rospy.get_param("~device", "cuda:0")
        with torch.inference_mode():
            initialize(version_base='1.3.2', config_path="../Cutie/cutie/config", job_name="eval_config")
            cfg = compose(config_name="eval_config")

            with open_dict(cfg):
                # TODO
                cfg['weights'] = rospy.get_param("~weights")
            data_cfg = get_dataset_cfg(cfg)

            # Load the network weights
            cutie = CUTIE(cfg).cuda().eval()
            model_weights = torch.load(cfg.weights)
            cutie.load_weights(model_weights)

        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber(
            "~input_image",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.pub_vis_img = rospy.Publisher("~output_image", Image, queue_size=1)
        self.pub_segmentation_img = rospy.Publisher(
            "~segmentation", Image, queue_size=1
        )

        self.prediction = None
        input_seg_msg = rospy.wait_for_message("~input_segmentation", Image)
        self.template_mask = self.bridge.imgmsg_to_cv2(input_seg_msg, desired_encoding="32SC1")
        input_img_msg = rospy.wait_for_message("~input_image", Image)
        self.image = self.bridge.imgmsg_to_cv2(input_img_msg, desired_encoding="rgb8")
        self.num_mask = len(np.unique(self.template_mask)) - 1

        torch.cuda.empty_cache()
        self.processor = InferenceCore(cutie, cfg=cfg)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                frame_torch = image_to_torch(self.image, device=self.device)
                # initialize with the mask
                mask_torch = index_numpy_to_one_hot_torch(self.template_mask, self.num_mask+1).to(self.device)
                # the background mask is not fed into the model
                #
                self.prediction = self.processor.step(frame_torch, mask_torch[1:], idx_mask=False)

    @torch.inference_mode()
    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        if self.prediction is not None:
            with torch.cuda.amp.autocast(enabled=True):

                # convert numpy array to pytorch tensor format
                frame_torch = image_to_torch(self.image, device=self.device)
                # propagate only
                prediction = self.processor.step(frame_torch)

                # argmax, convert to numpy
                self.prediction = torch_prob_to_numpy_mask(prediction)
                self.visualization = overlay_davis(self.image, self.prediction)

                seg_msg = self.bridge.cv2_to_imgmsg(
                    self.prediction.astype(np.int32), encoding="32SC1"
                )
                seg_msg.header.stamp = rospy.Time.now()
                seg_msg.header.frame_id = img_msg.header.frame_id
                self.pub_segmentation_img.publish(seg_msg)

                vis_img_msg = self.bridge.cv2_to_imgmsg(self.visualization, encoding="rgb8")
                vis_img_msg.header.stamp = rospy.Time.now()
                vis_img_msg.header.frame_id = img_msg.header.frame_id
                self.pub_vis_img.publish(vis_img_msg)


if __name__ == "__main__":
    rospy.init_node("cutie_node")
    node = CutieNode()
    rospy.spin()
