# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

from pupil_apriltags import Detector, Detection
import numpy as np
import quaternion
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

class TagInfo:
    def __init__(self, id:int, t:np.ndarray, q:np.ndarray):
        self.id = id
        self.t = t
        self.q = q


class ApriltagNode(Node):
    def __init__(self):
        super().__init__("apriltag")

        self.params_tag:list[rclpy.Parameter] = []
        self.params_tag.append(self.declare_parameter(
            "family", "tag36h11",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="apriltags family",
                read_only=True
            )
        ))
        self.params_tag.append(self.declare_parameter(
            "id", [0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description="ID of each tag",
                read_only=True
            )
        ))
        self.params_tag.append(self.declare_parameter(
            "tx", [0.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="x translation of each tag"
            )
        ))
        self.params_tag.append(self.declare_parameter(
            "ty", [0.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="y translation of each tag"
            )
        ))
        self.params_tag.append(self.declare_parameter(
            "tz", [0.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="z translation of each tag"
            )
        ))
        self.params_tag.append(self.declare_parameter(
            "qw", [1.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="w rotation of each tag"
            )
        ))
        self.params_tag.append(self.declare_parameter(
            "qx", [0.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="x rotation of each tag"
            )
        ))
        self.params_tag.append(self.declare_parameter(
            "qy", [0.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="y rotation of each tag"
            )
        ))
        self.params_tag.append(self.declare_parameter(
            "qz", [0.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="z rotation of each tag"
            )
        ))

        fam = self.params_tag[0].get_parameter_value().string_value
        self.get_logger().info("fam={}".format(fam))
        self.detector = Detector(families=fam)

        tag_id = self.params_tag[1].get_parameter_value().integer_array_value
        self.get_logger().info("id={}".format(tag_id))

        n_tag = len(tag_id)

        t = []
        for i in range(2,5):
            t.append(np.array(self.params_tag[i].get_parameter_value().double_array_value))
        t = np.stack(t).T
        self.get_logger().info("t=\n{}".format(t))

        q = []
        for i in range(5,9):
            q.append(np.array(self.params_tag[i].get_parameter_value().double_array_value,))
        q = np.stack(q).T
        self.get_logger().info("q=\n{}".format(q))

        self.tags:list[TagInfo] = []
        for i in range(n_tag):
            self.tags.append(TagInfo(tag_id[i], t[i], q[i]))
            self.get_logger().info("tag[{}]=({}, {})".format(i, self.tags[i].t, self.tags[i].q))

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_tag_tf()

        self.sub_req_tf = self.create_subscription(Bool, "req_apriltag_tf", self.cb_req_tf, 1)
        self.sub_image = self.create_subscription(Image, "image", self.cb_image, 10)


    def broadcast_tag_tf(self):
        n_tag = len(self.tags)
        tfs = []
        for i in range(n_tag):
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = "world"
            tf.child_frame_id = "apriltag{}".format(self.tags[i].id)

            tf.transform.translation.x = float(self.tags[i].t[0])
            tf.transform.translation.y = float(self.tags[i].t[1])
            tf.transform.translation.z = float(self.tags[i].t[2])
            tf.transform.rotation.w = float(self.tags[i].q[0])
            tf.transform.rotation.x = float(self.tags[i].q[1])
            tf.transform.rotation.y = float(self.tags[i].q[2])
            tf.transform.rotation.z = float(self.tags[i].q[3])

            tfs.append(tf)

        self.static_tf_broadcaster.sendTransform(tfs)

        self.get_logger().info("broadcasted tf")


    def cb_req_tf(self, msg:Bool):
        self.broadcast_tag_tf()


    def cb_image(self, msg:Image):
        pass
        # img = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
        # detections:list[Detection] = self.detector.detect(img, estimate_tag_pose=True, camera_params=)
        # for detection in detections:
        #     _t:np.ndarray = detection.pose_t.astype(np.float32)
        #     _t[2] *= -1
        #     _R:np.ndarray = 

        # t = TransformStamped()

        # t.header.stamp = self.get_clock().now().to_msg()
        # t.header.frame_id = "world"
        # t.child_frame_id = "tag{}".format()


def main():
    rclpy.init()
    node = ApriltagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
