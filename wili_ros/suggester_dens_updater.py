# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from wilitools import Gaussian, Suggester

from ._convert import msg_to_suggester, msg_to_hmm, dens_to_msg

class SuggesterDensUpdater(Node):
    def __init__(self):
        super().__init__("suggester_dens_updater")

        self.suggester:Suggester = None

        self.pub_new_dens = self.create_publisher(Float32MultiArray, "new_dens", 1)
        self.pub_suggest_res = self.create_publisher(Float32MultiArray, "suggest_res", 5)

        self.sub_init_suggester = self.create_subscription(Float32MultiArray, "init_suggester", self.cb_init_suggester, 1)
        self.sub_new_hmm = self.create_subscription(Float32MultiArray, "new_hmm", self.cb_new_hmm, 1)
        self.sub_where_found = self.create_subscription(Float32MultiArray, "where_found", self.cb_where_found, 5)
        self.sub_suggest_req = self.create_publisher(Float32MultiArray, "suggest_req", self.cb_suggest_req, 5)


    def cb_init_suggester(self, msg:Float32MultiArray):
        init_prob, tr_prob, avrs, covars, miss_probs, dens_miss_probs = msg_to_suggester(msg)
        self.suggester = Suggester(
            init_prob, tr_prob, avrs, covars, 
            miss_probs, dens_miss_probs
        )


    def cb_new_hmm(self, msg:Float32MultiArray):
        if self.suggester is None:
            return

        init_prob, tr_prob, avrs, covars = msg_to_hmm(msg)
        self.suggester.init_prob = init_prob
        self.suggester.tr_prob = tr_prob
        self.suggester.gaussian.avrs = avrs
        self.suggester.gaussian.covars = covars
        self.get_logger("subscribed \"new_hmm\"")


    def cb_where_found(self, msg:Float32MultiArray):
        if self.suggester is None:
            return
        
        self.suggester.update(np.array(msg.data, dtype=np.float32))

        msg = dens_to_msg(self.suggester.dens_miss_probs)
        self.pub_new_dens.publish(msg)


    def cb_suggest_req(self, msg:Float32MultiArray):
        if Suggester is None:
            dim = MultiArrayDimension(label="none", size=0, stride=0)
            layout = MultiArrayLayout(dim=[dim], data_offset=0)
            msg = Float32MultiArray(layout=layout, data=[])
            self.pub_suggest_res.publish(msg)
            return
        
        shape = []
        for d in msg.layout.dim:
            shape.append(d.size)

        x = np.array(msg.data, dtype=np.float32).reshape(shape)
        h = self.suggester.suggest(x)
        if type(h) == np.float32:
            h = np.array([h], dtype=np.float32)

        dims = []
        stride = 1
        for d in msg.layout.dim[:0:-1]:
            stride *= d.size
            dims.append(MultiArrayDimension(label=d.label, size=d.size, stride=stride))
        dims = dims[::-1]

        layout = MultiArrayLayout(dim=dims, data_offset=0)
        msg = Float32MultiArray(layout=layout, data=h.flatten())
        self.pub_suggest_res(msg)


def main():
    rclpy.init()
    node = SuggesterDensUpdater()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
