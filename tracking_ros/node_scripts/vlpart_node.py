#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import supervision as sv
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import Label, LabelArray
from jsk_recognition_msgs.msg import ClassificationResult
from tracking_ros_utils.srv import SamPrompt, SamPromptRequest

from model_config import SAMConfig, VLPartConfig
from utils import overlay_davis

# from dynamic_reconfigure.server import Server
# from tracking_ros.cfg import VLPartConfig as ServerConfig

BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

class VLPartNode(ConnectionBasedTransport):
    def __init__(self):
        super(VLPartNode, self).__init__()
        # self.reconfigure_server = Server(ServerConfig, self.config_cb)
        self.vocabulary = rospy.get_param("~vocabulary", "custom")
        self.classes = [_class.strip() for _class in rospy.get_param("~classes", "bottle cap; cup handle;").split(";") if _class.strip()]
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        self.use_sam = rospy.get_param("~use_sam", False)
        self.initialize()

        self.bridge = CvBridge()
        self.pub_vis_img = self.advertise("~output/debug_image", Image, queue_size=1)
        self.pub_rects = self.advertise("~output/rects", RectArray, queue_size=1)
        self.pub_labels = self.advertise("~output/labels", LabelArray, queue_size=1)
        self.pub_class = self.advertise("~output/class", ClassificationResult, queue_size=1)
        self.pub_seg = self.advertise("~output/segmentation", Image, queue_size=1)

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

    def initialize(self):
        self.detect_flag = False
        self.vlpart_config = VLPartConfig.from_rosparam()
        self.vlpart_predictor = self.vlpart_config.get_predictor(
            vocabulary=self.vocabulary,
            custom_vocabulary=self.classes,
            confidence_threshold=self.confidence_threshold,
        )
        # initialize the model with the mask
        self.detect_flag = True

    # TODO: Detectron2 does not support modifying metadata while running
    # def config_cb(self, config, level):
    #     self.vocabulary = config.vocabulary
    #     self.classes = [_class.strip() for _class in config.classes.split(";") if _class.strip()]
    #     self.confidence_threshold = config.confidence_threshold
    #     self.initialize()
    #     return config

    def publish_result(self, boxes, label_names, scores, mask, vis, frame_id):
        if label_names is not None:
            label_array = LabelArray()
            label_array.labels = [Label(id=i + 1, name=name) for i, name in enumerate(label_names)]
            label_array.header.stamp = rospy.Time.now()
            label_array.header.frame_id = frame_id
            self.pub_labels.publish(label_array)

            class_result = ClassificationResult(
                header=label_array.header,
                classifier=self.vlpart_config.model_name,
                target_names=self.classes,
                labels=[self.classes.index(name) for name in label_names],
                label_names=label_names,
                label_proba=scores,
            )
            self.pub_class.publish(class_result)

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
            rect_array.header.stamp = rospy.Time.now()
            rect_array.header.frame_id = frame_id
            self.pub_rects.publish(rect_array)

        if vis is not None:
            vis_img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="rgb8")
            vis_img_msg.header.stamp = rospy.Time.now()
            vis_img_msg.header.frame_id = frame_id
            self.pub_vis_img.publish(vis_img_msg)

        if mask is not None:
            seg_msg = self.bridge.cv2_to_imgmsg(mask, encoding="32SC1")
            seg_msg.header.stamp = rospy.Time.now()
            seg_msg.header.frame_id = frame_id
            self.pub_seg.publish(seg_msg)

    def callback(self, img_msg):
        if self.detect_flag:
            self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
            self.segmentation = None
            self.visualization = self.image.copy()
            xyxys = None
            labels = None
            scores = None

            # vlpart model inference
            predictions, _ = self.vlpart_predictor.run_on_image(cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            instances = predictions['instances'].to('cpu')
            classes = instances.pred_classes.tolist()

            if classes: # if there are any detections
                boxes = instances.pred_boxes.tensor
                scores = instances.scores
                masks = instances.pred_masks # [N, H, W]
                for i, mask in enumerate(masks):
                    if self.segmentation is None:
                        self.segmentation = mask.numpy().astype(np.int32)
                    else:
                        self.segmentation[mask] = i + 1
                self.visualization = overlay_davis(self.visualization, self.segmentation)

                labels = [self.classes[cls_id] for cls_id in classes]
                scores = scores.tolist()
                labels_with_scores = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]
                xyxys = boxes.cpu().numpy() # [N, 4]

                detections = sv.Detections(
                    xyxy=xyxys,
                    class_id=np.array(classes),
                    confidence=np.array(scores),
                )
                if self.use_sam:
                    rospy.wait_for_service("/sam_node/process_prompt")
                    try:
                        prompt = SamPromptRequest()
                        prompt.image = img_msg
                        for xyxy in detections.xyxy:
                            rect = Rect()
                            rect.x = int(xyxy[0])  # x1
                            rect.y = int(xyxy[1])  # y1
                            rect.width = int(xyxy[2] - xyxy[0])  # x2 - x1
                            rect.height = int(xyxy[3] - xyxy[1])  # y2 - y1
                            prompt.boxes.append(rect)
                        prompt_response = rospy.ServiceProxy("/sam_node/process_prompt", SamPrompt)
                        res = prompt_response(prompt)
                        seg_msg, vis_img_msg = res.segmentation, res.segmentation_image
                        self.segmentation = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding="32SC1")
                        self.visualization = self.bridge.imgmsg_to_cv2(vis_img_msg, desired_encoding="rgb8")
                    except rospy.ServiceException as e:
                        rospy.logerr(f"Service call failed: {e}")
                self.visualization = BOX_ANNOTATOR.annotate(scene=self.visualization, detections=detections)
                self.visualization = LABEL_ANNOTATOR.annotate(
                    scene=self.visualization, detections=detections, labels=labels_with_scores
                )
            self.publish_result(
                xyxys, labels, scores, self.segmentation, self.visualization, img_msg.header.frame_id
            )



if __name__ == "__main__":
    rospy.init_node("vlpart_node")
    node = VLPartNode()
    rospy.spin()
