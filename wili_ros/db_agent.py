# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Bool
from wilitools import Gaussian, wiliDB

from ._convert import suggester_to_msg, msg_to_hmm, msg_to_dens

class DBAgent(Node):
    def __init__(self):
        super().__init__("db_agent")

        param_db_url = self.declare_parameter("db_url")
        param_area_id = self.declare_parameter("area_id")

        self.db_url = param_db_url.get_parameter_value().string_value
        self.area_id = param_area_id.get_parameter_value().integer_value

        self.get_logger().info("db_url={}".format(self.db_url))
        self.get_logger().info("area_id={}".format(self.area_id))

        self.db = wiliDB(self.db_url)
        self.get_logger().info("connected with database".format(self.area_id))

        self.pub_init_suggester = self.create_publisher(Float32MultiArray, "init_suggester", 1)

        self.sub_req_init_suggester = self.create_subscription(Bool, "req_init_suggester", self.cb_req_init_suggester, 1)
        self.sub_new_hmm = self.create_subscription(Float32MultiArray, "new_hmm", self.cb_new_hmm, 1)
        self.sub_new_dens = self.create_subscription(Float32MultiArray, "new_dens", self.cb_new_dens, 1)


    def cb_req_init_suggester(self, msg:Bool):
        init_prob = self.db.read_init_prob(self.area_id)
        tr_prob = self.db.read_tr_prob(self.area_id)
        gaussian = self.db.read_gaussian(self.area_id)
        miss_probs, dens_miss_probs = self.db.read_samples(self.area_id)
        pub_msg = suggester_to_msg(
            init_prob, tr_prob, gaussian.avrs, gaussian.covars,
            miss_probs, dens_miss_probs
        )
        self.pub_init_suggester.publish(pub_msg)

        self.get_logger().info("these are params in db")


    def cb_new_hmm(self, msg:Float32MultiArray):
        init_prob, tr_prob, avrs, covars = msg_to_hmm(msg)
        self.db.update_init_prob(self.area_id, init_prob)
        self.db.update_tr_prob(self.area_id, tr_prob)
        self.db.update_gaussian(self.area_id, Gaussian(avrs, covars))
        self.get_logger().info("updated hmm params in db")


    def cb_new_dens(self, msg:Float32MultiArray):
        dens_miss_probs = msg_to_dens(msg)
        self.db.update_dens(self.area_id, dens_miss_probs)
        self.get_logger().info("updated miss pobs in db")


def main():
    rclpy.init()
    node = DBAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
