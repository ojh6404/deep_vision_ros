#!/usr/bin/env python

from argparse import ArgumentParser
import os

import numpy as np
import torch

import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.eval_args import add_common_eval_args
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.ext.grounding_dino import segment_with_text
try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel

from gui.interactive_utils import overlay_davis

torch.autograd.set_grad_enabled(False)

class DevaNode(object):
    def __init__(self):
        super(DevaNode, self).__init__()
        self.device = rospy.get_param("~device", "cuda:0")
        self.sam_model_type = rospy.get_param("~sam_model_type", "vit_t")

        self.data_path = os.path.join(rospkg.RosPack().get_path('tracking_ros'), 'trained_data')
        self.deva, self.gd, self.sam, self.cfg = self.init_model()
        self.classes = rospy.get_param("~classes")
        self.cfg['prompt'] = '.'.join(self.classes)

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
        self.ti = 0

    # TODO move to node config
    def init_model(self):

        # default parameters
        parser = ArgumentParser()
        add_common_eval_args(parser)
        add_ext_eval_args(parser)
        add_text_default_args(parser)
        args = parser.parse_args([])

        # grounding dino model
        args.GROUNDING_DINO_CONFIG_PATH = os.path.join(self.data_path, "groundingdino/GroundingDINO_SwinT_OGC.py")
        args.GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(self.data_path, "groundingdino/groundingdino_swint_ogc.pth")
        gd = GroundingDINOModel(model_config_path=args.GROUNDING_DINO_CONFIG_PATH,
                                    model_checkpoint_path=args.GROUNDING_DINO_CHECKPOINT_PATH,
                                    device=self.device)

        # sam model
        args.SAM_ENCODER_VERSION = self.sam_model_type
        if "hq" in args.SAM_ENCODER_VERSION:
            from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            sam_checkpoint = os.path.join(self.data_path, "sam", self.sam_model_type + ".pth")
            sam_model = sam_model_registry[self.sam_model_type](checkpoint=sam_checkpoint)
        elif args.SAM_ENCODER_VERSION == "vit_t":
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            sam_checkpoint = os.path.join(self.data_path, "sam", "mobile_sam.pth")
        else:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            sam_checkpoint = os.path.join(self.data_path, "sam", self.sam_model_type + ".pth")
        sam_model = sam_model_registry[self.sam_model_type](checkpoint=sam_checkpoint)
        sam_model.to(device=self.device)
        sam_model.eval()
        sam = SamPredictor(sam_model)

        # deva model
        args.model = os.path.join(self.data_path, "deva/DEVA-propagation.pth")

        cfg = vars(args)
        cfg['enable_long_term'] = True

        # Load our checkpoint
        deva_model = DEVA(cfg).cuda().eval()
        if args.model is not None:
            model_weights = torch.load(args.model)
            deva_model.load_weights(model_weights)
        else:
            rospy.logwarn('No model loaded.')

        cfg['enable_long_term_count_usage'] = True
        cfg['max_num_objects'] = 50
        cfg['amp'] = True
        cfg['chunk_size'] = 4
        cfg['detection_every'] = 5
        cfg['max_missed_detection_count'] = 10
        cfg['temporal_setting'] = 'online'
        cfg['pluralize'] = True
        cfg['DINO_THRESHOLD'] = 0.5


        deva = DEVAInferenceCore(deva_model, config=cfg)
        deva.next_voting_frame = cfg['num_voting_frames'] - 1
        deva.enabled_long_id()

        return deva, gd, sam, cfg

    def publish_result(self, mask, vis, frame_id):
        seg_msg = self.bridge.cv2_to_imgmsg(
            self.prediction.astype(np.int32), encoding="32SC1"
        )
        seg_msg.header.stamp = rospy.Time.now()
        seg_msg.header.frame_id = frame_id
        self.pub_segmentation_img.publish(seg_msg)

        vis_img_msg = self.bridge.cv2_to_imgmsg(self.visualization, encoding="rgb8")
        vis_img_msg.header.stamp = rospy.Time.now()
        vis_img_msg.header.frame_id = frame_id
        self.pub_vis_img.publish(vis_img_msg)


    @torch.inference_mode()
    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            min_size = min(self.image.shape[:2])
            deva_input_image = get_input_frame_for_deva(self.image, min_size)
            if self.ti % self.cfg["detection_every"] == 0:
                mask, segments_info = segment_with_text(self.cfg, self.gd, self.sam, self.image, self.classes, min_size)
                prob = self.deva.incorporate_detection(deva_input_image, mask, segments_info)
            else:
                prob = self.deva.step(deva_input_image, None, None)
            self.prediction = torch.argmax(prob, dim=0).cpu().numpy()
            self.visualization = overlay_davis(self.image, self.prediction)
            self.publish_result(self.prediction, self.visualization, img_msg.header.frame_id)
            self.ti += 1


if __name__ == "__main__":
    rospy.init_node("deva_node")
    node = DevaNode()
    rospy.spin()
