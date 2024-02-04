# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from wilitools import Gaussian, wiliDB

from ._convert import hmm_parameter_to_ros_msg, ros_msg_to_hmm_parameter

class DBAgent(Node):
    def __init__(self, db_path:str, area_id:int):
        super().__init__("db_agent")

        self.db = wiliDB(db_path)
        self.area_id = area_id

        self.pub_init_hmm = self.create_publisher(Float32MultiArray, 'init_hmm', 1)
        self.sub_new_hmm = self.create_subscription(Float32MultiArray, "new_hmm", self.cb_new_hmm, 1)

        init_prob = self.db.read_init_prob(area_id)
        tr_prob = self.db.read_tr_prob(area_id)
        gaussian = self.db.read_gaussian(area_id)
        msg = hmm_parameter_to_ros_msg(init_prob, tr_prob, gaussian.avrs, gaussian.covars)
        self.pub_init_hmm.publish(msg)
        self.get_logger("published \"init_hmm\"")


    def cb_new_hmm(self, msg:Float32MultiArray):
        init_prob, tr_prob, avrs, covars = ros_msg_to_hmm_parameter(msg)
        self.db.update_init_prob(self.area_id, init_prob)
        self.db.update_tr_prob(self.area_id, tr_prob)
        self.db.update_gaussian(self.area_id, Gaussian(avrs, covars))
        self.get_logger("subscribed \"new_hmm\"")


def main():
    rclpy.init()
    node = DBAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
