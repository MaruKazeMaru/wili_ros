# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension

class ObsStacker(Node):
    def __init__(self):
        super().__init__("obs_stacker")

        self.obs_max_len = 100

        self.param_obs_len = self.declare_parameter("observation_len")
        self.obs_min_len = self.param_obs_len.get_parameter_value().integer_value

        self.pub_observation = self.create_publisher(Float32MultiArray, "observation", 5)

        self.sub_observation_one = self.create_subscription(Float32MultiArray, "observation_one", self.cb_observation_one, 10)

        self.obs:np.ndarray = None
        self.obs_cnt = 0


    def cb_observation_one(self, msg:Float32MultiArray):
        # stack observation_one
        msg_ = np.array(msg.data[:2], dtype=np.float32).reshape((1,2))
        if self.obs is None:
            self.obs = msg_
        else:
            self.obs = np.concatenate([self.obs, msg_], axis=0)
        k = self.obs.shape[0]
        self.get_logger().info("stacked {}th obs".format(k+1))

        # try publish
        if np.random.randint(0,2) == 1:
            self.obs_cnt += 1
        if (self.obs_cnt >= self.obs_min_len) or (k >= self.obs_max_len):
            dims = [
                MultiArrayDimension(label="observation", size=k, stride=2),
                MultiArrayDimension(label="xy", size=2, stride=1),
            ]
            layout = MultiArrayLayout(dim=dims, data_offset=0)
            pub_msg = Float32MultiArray(layout=layout, data=self.obs.flatten())
            self.pub_observation.publish(pub_msg)

            # reset obs
            self.obs_cnt = 0
            self.obs = None
            self.get_logger().info("flushed obs")


def main():
    rclpy.init()
    node = ObsStacker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
