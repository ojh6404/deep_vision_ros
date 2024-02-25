#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
import torchvision
import supervision as sv

import rospy
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from sensor_msgs.msg import Image

from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import Label, LabelArray

from deva.dataset.utils import im_normalization
from deva.inference.object_info import ObjectInfo

from tracking_ros.cfg import GroundingDINOConfig as ServerConfig
from model_config import SAMConfig, GroundingDINOConfig, DEVAConfig
from utils import overlay_davis

torch.autograd.set_grad_enabled(False)

BOX_ANNOTATOR = sv.BoxAnnotator()

class DevaNode(ConnectionBasedTransport):
    def __init__(self):
        super(DevaNode, self).__init__()
        self.sam_config = SAMConfig.from_rosparam()
        self.gd_config = GroundingDINOConfig.from_rosparam()
        self.deva_config = DEVAConfig.from_rosparam()
        self.reconfigure_server = Server(ServerConfig, self.config_cb)
        self.with_bbox = rospy.get_param("~with_bbox", True)

        self.bridge = CvBridge()
        self.pub_segmentation_img = self.advertise("~output/segmentation", Image, queue_size=1)
        self.pub_vis_img = self.advertise("~output/segmentation_image", Image, queue_size=1)
        self.pub_rects = self.advertise("~output/rects", RectArray, queue_size=1)
        # self.pub_labels = self.advertise("~output/labels", LabelArray, queue_size=1)
        # self.pub_class = self.advertise("~output/class", ClassificationResult, queue_size=1)

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

    def config_cb(self, config, level):
        self.track_flag = False
        self.sam_predictor = self.sam_config.get_predictor()
        self.gd_predictor = self.gd_config.get_predictor()
        self.deva_predictor, self.cfg = self.deva_config.get_predictor()  # TODO integrate cfg into DEVAConfig
        self.classes = [_class.strip() for _class in config.classes.split(";")]
        self.cfg["prompt"] = ".".join(self.classes)
        self.cnt = 0
        self.track_flag = True
        return config

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
            with torch.cuda.amp.autocast(enabled=self.cfg["amp"]):
                torch_image = im_normalization(torch.from_numpy(self.image).permute(2, 0, 1).float() / 255)
                deva_input = torch_image.to(self.deva_config.device)
                if self.cnt % self.cfg["detection_every"] == 0:  # object detection query
                    self.sam_predictor.set_image(self.image, image_format="RGB")
                    # detect objects with GroundingDINO
                    detections = self.gd_predictor.predict_with_classes(
                        image=cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
                        classes=self.classes,
                        box_threshold=self.cfg["DINO_THRESHOLD"],
                        text_threshold=self.cfg["DINO_THRESHOLD"],
                    )
                    nms_idx = (
                        torchvision.ops.nms(
                            torch.from_numpy(detections.xyxy),
                            torch.from_numpy(detections.confidence),
                            self.cfg["DINO_NMS_THRESHOLD"],
                        )
                        .numpy()
                        .tolist()
                    )
                    detections.xyxy = detections.xyxy[nms_idx]
                    detections.confidence = detections.confidence[nms_idx]
                    detections.class_id = detections.class_id[nms_idx]
                    # segment objects with SAM
                    result_masks = []
                    for box in detections.xyxy:
                        masks, scores, _ = self.sam_predictor.predict(box=box, multimask_output=True)
                        index = np.argmax(scores)
                        result_masks.append(masks[index])
                    detections.mask = np.array(result_masks)
                    incorporate_mask = torch.zeros(
                        self.image.shape[:2], dtype=torch.int64, device=self.gd_predictor.device
                    )
                    curr_id = 1
                    segments_info = []
                    # sort by descending area to preserve the smallest object
                    for i in np.flip(np.argsort(detections.area)):
                        mask = detections.mask[i]
                        confidence = detections.confidence[i]
                        class_id = detections.class_id[i]
                        mask = torch.from_numpy(mask.astype(np.float32))
                        mask = (mask > 0.5).float()
                        if mask.sum() > 0:
                            incorporate_mask[mask > 0] = curr_id
                            segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence))
                            curr_id += 1
                    prob = self.deva_predictor.incorporate_detection(deva_input, incorporate_mask, segments_info)
                    self.cnt = 1
                else:
                    prob = self.deva_predictor.step(deva_input, None, None)
                    self.cnt += 1
                self.mask = torch.argmax(prob, dim=0).cpu().numpy()  # (H, W)
                object_num = len(np.unique(self.mask)) - 1
                if self.with_bbox and object_num > 0:
                    masks = []
                    for i in np.unique(self.mask)[1:]:
                        mask = (self.mask == i).astype(np.uint8)
                        masks.append(mask)
                    self.masks = np.stack(masks, axis=0)  # (N, H, W)
                    xyxy = sv.mask_to_xyxy(self.masks)
                    object_ids = np.unique(self.mask)[1:]  # without background
                    detections = sv.Detections(
                        xyxy=xyxy,
                        mask=self.masks,
                        class_id=object_ids,
                    )
                    painted_image = overlay_davis(self.image.copy(), self.mask)
                    # TODO convert labels to class name, but it needs some trick because object id and class id is not consistent between tracking and detecting
                    self.visualization = BOX_ANNOTATOR.annotate(
                        scene=painted_image,
                        detections=detections,
                        labels=[f"ObjectID: {obj_id}" for obj_id in object_ids],
                    )

                    rects = []
                    for box in xyxy:
                        rect = Rect()
                        rect.x = int((box[0] + box[2]) / 2)
                        rect.y = int((box[1] + box[3]) / 2)
                        rect.width = int(box[2] - box[0])
                        rect.height = int(box[3] - box[1])
                        rects.append(rect)
                    rect_array = RectArray(rects=rects)
                    rect_array.header.stamp = rospy.Time.now()
                    rect_array.header.frame_id = img_msg.header.frame_id
                    self.pub_rects.publish(rect_array)

                    # label_names = [self.classes[cls_id] for cls_id in detections.class_id]
                    # label_array = LabelArray()
                    # label_array.labels = [Label(id=i + 1, name=name) for i, name in enumerate(label_names)]
                    # label_array.header.stamp = rospy.Time.now()
                    # label_array.header.frame_id = img_msg.header.frame_id
                    # class_result = ClassificationResult(
                    #     header=label_array.header,
                    #     classifier=self.gd_config.model_name,
                    #     target_names=self.classes,
                    #     labels=[self.classes.index(name) for name in label_names],
                    #     label_names=label_names,
                    #     label_proba=detections.confidence.tolist(),
                    # )
                    # self.pub_labels.publish(label_array)
                    # self.pub_class.publish(class_result)
                else:
                    self.visualization = overlay_davis(self.image.copy(), self.mask)
                self.publish_result(self.mask, self.visualization, img_msg.header.frame_id)


if __name__ == "__main__":
    rospy.init_node("deva_node")
    node = DevaNode()
    rospy.spin()
