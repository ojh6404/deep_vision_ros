#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import jax
import jax.numpy as jnp

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from opencv_apps.msg import Point2D
from opencv_apps.msg import Point2DArrayStamped

from model_config import TAPNetConfig

# NOTE we should append tapnet to python path cause it's not a package based system though it is not clean
import sys
import rospkg
sys.path.insert(0, rospkg.RosPack().get_path("tracking_ros"))
from tapnet import tapir_model
from tapnet.utils import model_utils


# TODO move to config and make it configurable
NUM_POINTS = 8


def construct_initial_causal_state(num_points, num_resolutions):
    """Construct initial causal state."""
    value_shapes = {
        "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
    }
    fake_ret = {k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()}
    return [fake_ret] * num_resolutions * 4


class TAPNetNode(object):  # should not be ConnectionBasedNode cause tapnet tracker needs continuous input
    def __init__(self):
        super(TAPNetNode, self).__init__()
        self.tapnet_config = TAPNetConfig.from_rosparam()
        self.online_init_apply, self.online_predict_apply = self.tapnet_config.get_predictor()
        self.with_bbox = rospy.get_param("~with_bbox", False)

        self.bridge = CvBridge()
        self.initialize()

        self.pub_vis_img = rospy.Publisher("~output/debug_image", Image, queue_size=1)
        self.pub_point = rospy.Publisher("~output/point", Point2DArrayStamped, queue_size=1)
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

    def point_callback(self, point_msg):
        # if point x and point y is out of image shape, just pass
        point_x = int(point_msg.point.x)  # x is within 0 ~ width
        point_y = int(point_msg.point.y)  # y is within 0 ~ height
        if point_x < 1 or point_x > (self.image.shape[1] - 1) or point_y < 1 or point_y > (self.image.shape[0] - 1):
            rospy.logwarn("point {} is out of image shape".format([point_x, point_y]))
            return
        self.pos = (point_y, self.image.shape[1] - point_x)
        self.query_frame = True
        rospy.loginfo("query point {} added".format(self.pos))

    def initialize(self):
        # placeholder
        self.pos = tuple()
        self.query_frame = True
        self.have_point = [False] * NUM_POINTS
        self.query_features = None
        self.causal_state = None
        self.next_query_idx = 0

        # NOTE Call one time to compile
        input_img_msg = rospy.wait_for_message("~input_image", Image)
        self.image = self.bridge.imgmsg_to_cv2(input_img_msg, desired_encoding="bgr8")
        rospy.loginfo("Compiling jax functions (this may take a while...)")
        self.query_points = jnp.zeros([NUM_POINTS, 3], dtype=jnp.float32)
        self.query_features, _ = self.online_init_apply(
            frames=model_utils.preprocess_frames(self.image[None, None]),
            points=self.query_points[None, 0:1],
        )
        jax.block_until_ready(self.query_features)
        self.query_features, _ = self.online_init_apply(
            frames=model_utils.preprocess_frames(self.image[None, None]),
            points=self.query_points[None],
        )
        self.causal_state = construct_initial_causal_state(NUM_POINTS, len(self.query_features.resolutions) - 1)
        (self.prediction, self.causal_state), _ = self.online_predict_apply(
            frames=model_utils.preprocess_frames(self.image[None, None]),
            features=self.query_features,
            causal_context=self.causal_state,
        )
        jax.block_until_ready(self.prediction["tracks"])
        rospy.loginfo("jax functions compiled")

    def upd(self, s1, s2):
        return s1.at[:, self.next_query_idx : self.next_query_idx + 1].set(s2)

    def publish_result(self, points, vis, frame_id):
        if points is not None:
            point_msg = Point2DArrayStamped()
            point_msg.header.stamp = rospy.Time.now()
            point_msg.header.frame_id = frame_id
            point_msg.points = [Point2D(x=p[0], y=p[1]) for p in points]
            self.pub_point.publish(point_msg)
        if vis is not None:
            vis_img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="rgb8")
            vis_img_msg.header.stamp = rospy.Time.now()
            vis_img_msg.header.frame_id = frame_id
            self.pub_vis_img.publish(vis_img_msg)

    def callback(self, img_msg):
        self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        if self.query_frame:  # point to track is added
            self.query_points = jnp.array((0,) + self.pos, dtype=jnp.float32)
            self.init_query_features, _ = self.online_init_apply(
                frames=model_utils.preprocess_frames(self.image[None, None]),
                points=self.query_points[None, None],
            )
            self.init_causal_state = construct_initial_causal_state(1, len(self.query_features.resolutions) - 1)
            self.query_frame = False
            self.causal_state = jax.tree_map(self.upd, self.causal_state, self.init_causal_state)
            self.query_features = tapir_model.QueryFeatures(
                lowres=jax.tree_map(self.upd, self.query_features.lowres, self.init_query_features.lowres),
                hires=jax.tree_map(self.upd, self.query_features.hires, self.init_query_features.hires),
                resolutions=self.query_features.resolutions,
            )
            self.have_point[self.next_query_idx] = True
            self.next_query_idx = (self.next_query_idx + 1) % NUM_POINTS
        if self.pos:
            (self.prediction, self.causal_state), _ = self.online_predict_apply(
                frames=model_utils.preprocess_frames(self.image[None, None]),
                features=self.query_features,
                causal_context=self.causal_state,
            )
            self.track = self.prediction["tracks"][0, :, 0]
            self.occlusion = self.prediction["occlusion"][0, :, 0]
            self.expected_dist = self.prediction["expected_dist"][0, :, 0]
            self.visibles = model_utils.postprocess_occlusions(self.occlusion, self.expected_dist)
            self.track = np.round(self.track)
            for i in range(len(self.have_point)):
                if self.visibles[i] and self.have_point[i]:
                    cv2.circle(
                        self.image, (int(self.track[i, 0]), int(self.track[i, 1])), 5, (255, 0, 0), -1
                    )  # visulize the tracked point
                    if self.track[i, 0] < 16 and self.track[i, 1] < 16:
                        rospy.loginfo((i, self.next_query_idx))
        self.visualization = self.image.copy()[:, ::-1]
        self.publish_result(None, self.visualization, img_msg.header.frame_id) # TODO publish tracking points


if __name__ == "__main__":
    rospy.init_node("tapnet_node")
    node = TAPNetNode()
    rospy.spin()
