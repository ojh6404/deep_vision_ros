#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is used to process rosbag files and save the processed results without roscore.
"""

import argparse
import time

import supervision as sv

import cv2
import numpy as np
import rosbag
import rospy
from cv_bridge import CvBridge
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import Label, LabelArray
from jsk_recognition_msgs.msg import ClassificationResult
from sensor_msgs.msg import CompressedImage, Image
from tqdm import tqdm

from tracking_ros.model_config import YOLOConfig, SAMConfig
from tracking_ros.utils import overlay_davis
from segment_anything.utils.amg import remove_small_regions

BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

class PatchTimer(rospy.Time):
    # PatchTimer Time.now so we don't need to call rospy.init_node (so we don't need a roscore)
    # Also solves problems with /use_sim_time (simulated time)
    def __init__(self, secs=0, nsecs=0):
        super(rospy.Time, self).__init__(secs, nsecs)

    @staticmethod
    def now():
        # initialize with wallclock
        float_secs = time.time()
        secs = int(float_secs)
        nsecs = int((float_secs - secs) * 1000000000)
        return PatchTimer(secs, nsecs)

# for no roscore
rospy.Time = PatchTimer

bridge = CvBridge()

class YOLOModel(object):
    def __init__(self):
        self.yolo_config = YOLOConfig.from_args(model_id="yolov8x-worldv2.pt",
                                                device="cuda:0")
        self.predictor = self.yolo_config.get_predictor()
        self.setup()
        self.get_mask = True
        if self.get_mask:
            self.sam_config = SAMConfig.from_args(model_type="vit_t", mode="prompt", device="cuda:0")
            self.sam_predictor = self.sam_config.get_predictor()
            self.refine_mask = True
            if self.refine_mask:
                self.area_threshold = 400
                self.refine_mode = "holes"

    def setup(self):
        self.classes = ["check cloth", "cup", "bottle"]
        self.predictor.set_classes(self.classes)
        self.box_threshold = 0.5
        self.nms_threshold = 0.5

    def process_image(self, image):
        self.image = image.copy()
        results = self.predictor.predict(self.image, save=False, conf=self.box_threshold, iou=self.nms_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [results.names[cls_id] for cls_id in detections.class_id]
        scores = detections.confidence.tolist()
        labels_with_scores = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]

        visualization = self.image.copy()
        segmentation = None
        if self.get_mask and len(detections.xyxy) > 0:
            result_mask = None
            for i, box in enumerate(detections.xyxy):
                mask, logit = self.process_prompt(
                    points=None,
                    labels=None,
                    bbox=np.array([box[0], box[1], box[2], box[3]]),
                    multimask=False,
                )
                if result_mask is None:
                    result_mask = mask.astype(np.uint8)
                else:
                    result_mask[mask] = i + 1
            visualization = self.image.copy()
            if result_mask is not None:
                visualization = overlay_davis(visualization, result_mask)
            segmentation = result_mask.astype(np.int32)
        visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
        visualization = LABEL_ANNOTATOR.annotate(
            scene=visualization, detections=detections, labels=labels_with_scores
        )
        return detections.xyxy, labels, scores, segmentation, visualization

    def process_prompt(
        self,
        points=None,
        bbox=None,
        labels=None,
        mask_input=None,
        multimask: bool = True,
    ):
        self.sam_predictor.set_image(self.image)
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            mask_input=mask_input,  # TODO
            multimask_output=multimask,
        )  # [N, H, W], B : number of prompts, N : number of masks recommended
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)]

        if self.refine_mask:
            # refine mask using logit
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bbox,
                mask_input=logit[None, :, :],
                multimask_output=multimask,
            )
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)]
            mask, _ = remove_small_regions(mask, self.area_threshold, mode=self.refine_mode)
        return mask, logit

    def write_result(self, rosbag_writer, header, t, boxes, label_names, scores, mask, vis):
        if label_names is not None:
            label_array = LabelArray()
            label_array.labels = [Label(id=i + 1, name=name) for i, name in enumerate(label_names)]
            label_array.header = header
            rosbag_writer.write("/labels", label_array, t)

            class_result = ClassificationResult(
                header=header,
                classifier=self.yolo_config.model_name,
                target_names=self.classes,
                labels=[self.classes.index(name) for name in label_names],
                label_names=label_names,
                label_proba=scores,
            )
            rosbag_writer.write("/class", class_result, t)

        if boxes is not None:
            rects = []
            for box in boxes:
                rect = Rect()
                rect.x = int(box[0])  # x1
                rect.y = int(box[1])  # y1
                rect.width = int(box[2] - box[0])  # x2 - x1
                rect.height = int(box[3] - box[1])  # y2 - y1
                rects.append(rect)
            rect_array = RectArray(rects=rects)
            rect_array.header = header
            rosbag_writer.write("/rects", rect_array, t)

        if vis is not None:
            vis_img_msg = bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            vis_img_msg.header = header
            rosbag_writer.write("/vis_img", vis_img_msg, t)

        if mask is not None:
            seg_msg = bridge.cv2_to_imgmsg(mask, encoding="32SC1")
            seg_msg.header = header
            rosbag_writer.write("/seg", seg_msg, t)


def main(args):
    model = YOLOModel()

    if args.output.endswith(".bag"):
        with rosbag.Bag(args.output, 'w') as outbag:
            for topic, msg, t in tqdm(rosbag.Bag(args.rosbag).read_messages()):
                if topic == args.topic:
                    outbag.write(args.topic, msg, t)
                    if "Compressed" in msg._type:
                        image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    else:
                        image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    xyxy, labels, scores, segmentation, visualization = model.process_image(image)
                    model.write_result(outbag, msg.header, t, xyxy, labels, scores, segmentation, visualization)
    elif args.output.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        for topic, msg, t in tqdm(rosbag.Bag(args.rosbag).read_messages()):
            if topic == args.topic:
                if "Compressed" in msg._type:
                    image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                else:
                    image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                xyxy, labels, scores, segmentation, visualization = model.process_image(image)
                if out is None:
                    out = cv2.VideoWriter(args.output, fourcc, 20.0, (visualization.shape[1], visualization.shape[0]))
                    out.write(visualization)
                else:
                    out.write(visualization)
        if out is not None:
            out.release()
        else:
            raise ValueError("no image message found in the rosbag file")

    else:
        raise ValueError("output file should be either rosbag file or mp4 file")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bag", "--rosbag", type=str, default=None, help="rosbag file to process")
    parser.add_argument("-o", "--output", type=str, default="output.bag", help="output rosbag file or mp4 file")
    parser.add_argument("-t", "--topic", type=str, default="/kinect_head/rgb/image_rect_color", help="topic to process")
    args = parser.parse_args()

    main(args)
