#!/usr/bin/env python3
import os
import rospy
import rospkg
import cv2
import cv_bridge
import numpy as np
import torch
from PIL import Image

from sensor_msgs.msg import CompressedImage
from jsk_topic_tools import ConnectionBasedTransport
from torch import device

from segment_anything import sam_model_registry, SamPredictor
from track_anything_ros.tracker.base_tracker import BaseTracker
from track_anything_ros.utils.util import (
    download_checkpoint,
    download_checkpoint_from_google_drive,
)


import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

MASK_COLOR = 3
MASK_ALPHA = 0.9
POINT_COLOR_N = 8
POINT_COLOR_P = 50
POINT_ALPHA = 0.9
POINT_RADIUS = 5
CONTOUR_COLOR = 1
CONTOUR_WIDTH = 0

package_path = rospkg.RosPack().get_path("track_anything_ros")


SAM_CHECKPOINT = {
    "vit_h": os.path.join(package_path, "checkpoints/sam_vit_h_4b8939.pth"),
    "vit_b": os.path.join(package_path,"checkpoints/sam_vit_b_01ec64.pth")
}
SAM_CHECKPOINT_URL_DICT = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
XMEM_CHECKPOINT = "XMem-s012.pth"
XMEM_CHECKPOINT_URL = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)

def cv2pil(image):
    """OpenCV型 -> PIL型"""
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    """PIL型 -> OpenCV型"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


def load_image(image_cv2):
    # load image
    image_pil = cv2pil(image_cv2)  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_grounding_dino_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def generate_multi_mask(masks):
    template_mask = np.zeros_like(masks[0])
    # for i in range(1, len(self.masks)):
    for i, mask in enumerate(masks):
        template_mask = np.clip(
            template_mask + mask * (i + 1),
            0,
            i + 1,
        )
    assert len(np.unique(template_mask)) == (len(masks) + 1)
    return template_mask

class TrackNode(object):
    def __init__(self):
        super(TrackNode, self).__init__()

        model_dir = rospy.get_param("~model_dir")
        tracker_config_file = rospy.get_param("~tracker_config_file")
        # inpainter_config_file = rospy.get_param("~inpainter_config_file")
        model_type = rospy.get_param("~model_type", "vit_h")
        self.model_type = model_type
        xmem_checkpoint = download_checkpoint(
            XMEM_CHECKPOINT_URL, model_dir, XMEM_CHECKPOINT
        )
        self.device = rospy.get_param("~device", "cuda:0")
        self.sam = self.init_segment_anything()
        self.xmem = BaseTracker(
            xmem_checkpoint, tracker_config_file, device=self.device
        )
        grounded_checkpoint = os.path.join(package_path, "checkpoints","groundingdino_swint_ogc.pth")
        self.config_file = os.path.join(package_path, "config","GroundingDINO_SwinT_OGC.py")
        self.grounding_dino_model = load_grounding_dino_model(self.config_file, grounded_checkpoint, device=self.device)

        self.bridge = cv_bridge.CvBridge()

        self.text_prompt = "pr2 robot arm and bowl"
        self.box_threshold = 0.3
        self.text_threshold = 0.25

        self.get_first_image()

        # self.pub_debug_image = self.advertise("~output_image", CompressedImage, queue_size=1)
        self.pub_segmentation_image = rospy.Publisher(
            "~segmentation_mask", CompressedImage, queue_size=1
        )
        
        self.sub = rospy.Subscriber(
            "~input_image",
            CompressedImage,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )


        self.logits = []
        self.painted_image = []
        self.masks = []

        # for place holder init
        self.embedded_image = None
        self.image = None
        self.track_image = None
        self.mask = None
        self.logit = None
        # self.template_mask = None
        self.painted_image = None

    def init_segment_anything(self):
        """
        Initialize the segmenting anything with model_type in ['vit_b', 'vit_l', 'vit_h']
        """
        sam = sam_model_registry[self.model_type](
            checkpoint=SAM_CHECKPOINT[self.model_type]
        ).to(self.device)
        predictor = SamPredictor(sam)
        return predictor

    def get_first_image(self):
        while not rospy.is_shutdown():
            self.first_image = self.bridge.compressed_imgmsg_to_cv2(rospy.wait_for_message("~input_image", CompressedImage))
            image_pil, image = load_image(self.first_image)
            boxes_filt, pred_phrases = get_grounding_output(
                self.grounding_dino_model, image, self.text_prompt, self.box_threshold, self.text_threshold, device=self.device
            )
            first_rgb = cv2.cvtColor(self.first_image, cv2.COLOR_BGR2RGB)
            self.sam.set_image(first_rgb)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = self.sam.transform.apply_boxes_torch(
                boxes_filt, first_rgb.shape[:2]
            ).to(self.device)

            if transformed_boxes.size(0) == 0:
                continue

            masks, _, _ = self.sam.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )

            self.masks = masks.squeeze(1).cpu().numpy()
            self.template_mask = generate_multi_mask(self.masks)

            self.mask, self.logit, self.painted_image = self.xmem.track(
                frame=self.first_image, first_frame_annotation=self.template_mask
            )
            rospy.loginfo("Initialized mask")
            break


    def callback(self, img_msg):
        # self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        self.image = self.bridge.compressed_imgmsg_to_cv2(img_msg)

        if self.template_mask is not None:  # track start
            self.mask, self.logit, self.painted_image = self.xmem.track(self.image)

            print('mask shape', self.mask.shape)
            print('mask unique', np.unique(self.mask))
            # encoding should be 32SC1 for jsk_pcl_utils/LabelToClusterPointIndices
            seg_mask_msg = self.bridge.cv2_to_compressed_imgmsg(np.clip(self.mask * 255, 0, 255).astype(np.uint8))
            seg_mask_msg.header = img_msg.header
            seg_mask_msg.format = "mono8; jpeg compressed "
            self.pub_segmentation_image.publish(seg_mask_msg)
        else:  # init
            pass

        # out_img_msg = self.bridge.cv2_to_imgmsg(self.painted_image, encoding="bgr8")
        # out_img_msg = self.bridge.cv2_to_compressed_imgmsg(self.image)
        # out_img_msg.header = img_msg.header
        # out_img_msg.format = "bgr8; jpeg compressed bgr8"
        # self.pub_debug_image.publish(out_img_msg)


if __name__ == "__main__":
    rospy.init_node("track_node")
    node = TrackNode()
    rospy.spin()
