#!/usr/bin/env python
import numpy as np
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PolygonStamped
from std_srvs.srv import Empty, EmptyResponse
from jsk_topic_tools import ConnectionBasedTransport


from segment_anything import sam_model_registry, SamPredictor
from track_anything_ros.tracker.base_tracker import BaseTracker
from track_anything_ros.utils.util import (
    download_checkpoint,
)
from track_anything_ros.utils.painter import mask_painter, point_drawer, bbox_drawer

class TrackNode(ConnectionBasedTransport):
    def __init__(self):
        super(TrackNode, self).__init__()

        model_dir = rospy.get_param("~model_dir")
        tracker_config_file = rospy.get_param("~tracker_config")
        model_type = rospy.get_param("~model_type", "vit_b")

        sam_checkpoint = download_checkpoint("sam_"+ model_type, model_dir)
        xmem_checkpoint = download_checkpoint("xmem", model_dir)
        self.device = rospy.get_param("~device", "cuda:0")


        # sam
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        # xmem
        self.xmem = BaseTracker(
            xmem_checkpoint, tracker_config_file, device=self.device
        )


        self.toggle_prompt_label_service = rospy.Service(
            "/track_anything/toggle_label", Empty, self.toggle_prompt_label_callback
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
        self.reset_embed_service = rospy.Service(
            "/track_anything/reset_embed", Empty, self.reset_embed_callback
        )

        self.track_trigger_service = rospy.Service(
            "/track_anything/track_trigger", Empty, self.track_trigger_callback
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
        self.subs = [self.sub_image, self.sub_point, self.sub_bbox]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
        self.xmem.clear_memory()

    def clear_points_callback(self, srv):
        rospy.loginfo("Clear prompts")
        self.points.clear()
        self.labels.clear()
        self.bbox = None
        self.mask = None
        self.logit = None
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
        if self.mask is None:
            rospy.logwarn("No mask to add")
            self.points.clear()
            self.labels.clear()
            return res
        self.masks.append(self.mask)
        self.points.clear()
        self.labels.clear()
        self.bbox = None
        self.num_mask += 1
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

    def track_trigger_callback(self, srv):
        rospy.loginfo("Tracking start...")
        self.template_mask = self.compose_mask(self.masks)
        self.mask, self.logit = self.xmem.track(
            frame=self.image, first_frame_annotation=self.template_mask
        )
        res = EmptyResponse()
        return res

    def toggle_prompt_label_callback(self, srv):
        self.label_mode = not self.label_mode
        rospy.loginfo("Toggle prompt label to {}".format(self.label_mode))
        res = EmptyResponse()
        return res


    def point_callback(self, point_msg):
        # if point x and point y is out of image shape, just pass
        point_x = int(point_msg.point.x) # x is within 0 ~ width
        point_y = int(point_msg.point.y) # y is within 0 ~ height

        if point_x < 1 or point_x > (self.image.shape[1] -1 ) or point_y < 1 or point_y > (self.image.shape[0] -1):
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

        self.mask, self.logit = self.process_prompt(
            points=np.array(self.points),
            labels=np.array(self.labels),
            bbox=np.array(self.bbox) if self.bbox is not None else None,
            multimask=self.multimask,
        )
        # self.masks.append(self.mask)

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

        self.mask, self.logit = self.process_prompt(
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
        multimask:bool=True,
    ):
        """
        it is used in first frame
        return: mask, logit, painted image(mask+point)
        """
        assert len(points) == len(labels)
        prompts = dict(
            point_coords= points,
            point_labels= labels,
            box=bbox,
            mask_input= mask_input, # TODO
            multimask_output= multimask,
        )
        masks, scores, logits = self.predictor.predict(
            **prompts) # [N, H, W], B : number of prompts, N : number of masks recommended
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores)] # choose the best mask [H, W]

        # refine mask using logit
        prompts = dict(
            point_coords= points,
            point_labels= labels,
            box=bbox,
            mask_input= logit[None, :, :],
            multimask_output= multimask,
        )
        masks, scores, logits = self.predictor.predict(**prompts)
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
    rospy.init_node("track_node")
    node = TrackNode()
    rospy.spin()
