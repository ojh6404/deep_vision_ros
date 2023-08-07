#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
import numpy as np
import cv2

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Empty, EmptyResponse
from jsk_topic_tools import ConnectionBasedTransport
from torch import device

from track_anything_ros.segmentator.sam_segmentator import SAMSegmentator
from track_anything_ros.tracker.base_tracker import BaseTracker
from track_anything_ros.utils.painter import point_painter, mask_painter
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

SAM_CHECKPOINT_DICT = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
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



class TrackNode(ConnectionBasedTransport):
    def __init__(self):
        super(TrackNode, self).__init__()

        model_dir = rospy.get_param("~model_dir")
        tracker_config_file = rospy.get_param("~tracker_config_file")
        # inpainter_config_file = rospy.get_param("~inpainter_config_file")
        model_type = rospy.get_param("~model_type", "vit_b")

        sam_checkpoint = download_checkpoint(
            SAM_CHECKPOINT_URL_DICT[model_type],
            model_dir,
            SAM_CHECKPOINT_DICT[model_type],
        )
        xmem_checkpoint = download_checkpoint(
            XMEM_CHECKPOINT_URL, model_dir, XMEM_CHECKPOINT
        )
        self.device = rospy.get_param("~device", "cuda:0")
        self.sam = SAMSegmentator(sam_checkpoint, model_type, device=self.device)
        self.xmem = BaseTracker(
            xmem_checkpoint, tracker_config_file, device=self.device
        )

        self.clear_points_service = rospy.Service(
            "/track_anything/clear_points", Empty, self.clear_points_callback
        )
        self.clear_masks_service = rospy.Service(
            "/track_anything/clear_masks", Empty, self.clear_masks_callback
        )
        self.add_mask_service = rospy.Service(
            "/track_anything/add_mask", Empty, self.add_mask_callback
        )
        self.set_embed_service = rospy.Service(
            "/track_anything/set_embed", Empty, self.set_embed_callback
        )
        self.reset_embed_service = rospy.Service(
            "/track_anything/reset_embed", Empty, self.reset_embed_callback
        )

        self.track_trigger_service = rospy.Service(
            "/track_anything/track_trigger", Empty, self.track_trigger_callback
        )

        self.pub_debug_image = self.advertise("~output_image", Image, queue_size=1)
        self.pub_segmentation_image = self.advertise(
            "~segmentation_mask", Image, queue_size=1
        )
        # self.pub_original_image = self.advertise("~original_image", Image, queue_size=1)
        self.bridge = CvBridge()

        self.points = []
        self.labels = []
        self.multimask = True

        self.logits = []
        self.painted_image = []
        self.masks = []

        # for place holder init
        self.embedded_image = None
        self.image = None
        self.track_image = None
        self.mask = None
        self.logit = None
        self.template_mask = None
        self.painted_image = None

    def clear_points_callback(self, srv):
        rospy.loginfo("Clear points")
        self.points.clear()
        self.labels.clear()
        self.mask = None
        self.logit = None
        self.painted_image = self.prev_painted_image.copy()
        res = EmptyResponse()
        return res

    def clear_masks_callback(self, srv):
        rospy.loginfo("Clear masks")
        self.masks.clear()
        self.mask = None
        self.logit = None
        self.painted_image = None
        res = EmptyResponse()
        return res

    def add_mask_callback(self, srv):
        rospy.loginfo("Mask added")

        res = EmptyResponse()
        if self.mask is None:
            rospy.logwarn("No mask to add")
            self.points.clear()
            self.labels.clear()
            return res

        self.masks.append(self.mask)
        self.points.clear()
        self.labels.clear()
        return res

    def set_embed_callback(self, srv):
        assert self.embedded_image is None, "reset before embedding"
        rospy.loginfo("Embedding image for segmentation")
        self.embedded_image = self.image
        self.sam.set_image(self.image)
        res = EmptyResponse()
        return res

    def reset_embed_callback(self, srv):
        rospy.loginfo("Reset Embedding image")
        self.embedded_image = None
        self.sam.reset_image()
        res = EmptyResponse()
        return res

    def track_trigger_callback(self, srv):
        rospy.loginfo("Tracking start...")
        self.template_mask = self.generate_multi_mask(self.masks)
        self.mask, self.logit, self.painted_image = self.xmem.track(
            frame=self.image, first_frame_annotation=self.template_mask
        )
        res = EmptyResponse()
        return res

    def subscribe(self):
        self.sub_image = rospy.Subscriber(
            "~input_image",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.sub_point = rospy.Subscriber(
            "~input_point",
            PointStamped,
            self.point_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.subs = [self.sub_image, self.sub_point]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
        self.xmem.clear_memory()

    def point_callback(self, point_msg):
        # TODO: clipping point
        # point_x = np.clip(int(point_msg.point.x), 0, self.image.shape[0])
        # point_y = np.clip(int(point_msg.point.y), 0, self.image.shape[1])

        # point = [point_x, point_y]
        point = [int(point_msg.point.x), int(point_msg.point.y)]
        label = 1  # TODO: add negative label
        rospy.loginfo("point {} and label {} added".format(point, label))
        self.points.append(point)
        self.labels.append(label)

        if self.embedded_image is None:
            self.sam.set_image(self.image)
            self.embedded_image = self.image

        self.mask, self.logit = self.sam.process_prompt(
            image=self.image,
            points=np.array(self.points),
            labels=np.array(self.labels),
            multimask=self.multimask,
        )

        self.prev_painted_image = self.image.copy()

    def generate_multi_mask(self, masks):
        template_mask = np.zeros_like(masks[0])
        # for i in range(1, len(self.masks)):
        for i, mask in enumerate(masks):
            template_mask = np.clip(
                template_mask + mask * (i + 1),
                0,
                i + 1,
            )

        assert len(np.unique(template_mask)) == (len(self.masks) + 1)
        return template_mask

    def change_mask_hsv(self, mask, original_image):
        orig_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        color = 100
        change_hsv = orig_hsv.copy()
        change_hsv[mask > 0, 0] = color
        return cv2.cvtColor(change_hsv, cv2.COLOR_HSV2BGR)

    def change_mask_rgb(self, mask, original_image):
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        color = [255, 0, 0]
        change_rgb = original_image.copy()
        change_rgb[mask > 0] = color
        return cv2.cvtColor(change_rgb, cv2.COLOR_RGB2BGR)

    def crop_image(self, image, width, height, x, y):
        return image[y : y + height, x : x + width]

    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        if self.template_mask is not None:  # track start
            self.mask, self.logit, self.painted_image = self.xmem.track(self.image)

            seg_mask = self.bridge.cv2_to_imgmsg(
                self.mask.astype(np.uint8), encoding="mono8"
            )

            seg_mask.header = img_msg.header
            # seg_mask.header.stamp = rospy.Time.now()
            self.pub_segmentation_image.publish(seg_mask)

        else:  # init
            if self.mask is not None:
                self.painted_image = mask_painter(
                    self.painted_image,
                    self.mask.astype("uint8"),
                    20 + MASK_COLOR,
                    MASK_ALPHA,
                    CONTOUR_COLOR,
                    CONTOUR_WIDTH,
                )

            # if len(self.masks) > 0:
            for i, mask in enumerate(self.masks):
                self.painted_image = mask_painter(
                    self.painted_image,
                    mask.astype("uint8"),
                    i + MASK_COLOR,
                    MASK_ALPHA,
                    CONTOUR_COLOR,
                    CONTOUR_WIDTH,
                )

            self.painted_image = point_painter(
                self.image if self.painted_image is None else self.painted_image,
                np.array(self.points),
                len(self.masks) + POINT_COLOR_N,
                POINT_ALPHA,
                POINT_RADIUS,
                CONTOUR_COLOR,
                CONTOUR_WIDTH,
            )

        debug_img_msg = self.bridge.cv2_to_imgmsg(self.painted_image, encoding="bgr8")
        debug_img_msg.header = img_msg.header
        # debug_img_msg.header.stamp = rospy.Time.now()
        self.pub_debug_image.publish(debug_img_msg)


if __name__ == "__main__":
    rospy.init_node("track_node")
    node = TrackNode()
    rospy.spin()
