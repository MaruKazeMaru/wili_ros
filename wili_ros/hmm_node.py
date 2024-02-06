# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

from hmmlearn.hmm import GaussianHMM
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool

from ._convert import hmm_to_msg, msg_to_suggester

class HMMNode(Node):
    def __init__(self):
        super().__init__('hmm')

        self.motion_num:int = None
        self.model:GaussianHMM = None

        self.pub_req_init_suggester = self.create_publisher(Bool, "req_init_suggester", 1)
        self.pub_new_hmm = self.create_publisher(Float32MultiArray, "new_hmm", 1)

        self.sub_init_suggester = self.create_subscription(Float32MultiArray, "init_suggester", self.cb_init_suggester, 1)
        self.sub_obs = self.create_subscription(Float32MultiArray, "observation", self.cb_observation, 5)

        self.timer_req_init_suggester = self.create_timer(2, self.cb_req_init_suggester)


    def cb_req_init_suggester(self):
        msg = Bool()
        self.pub_req_init_suggester.publish(msg)
        self.get_logger().info("I want to init params")


    def cb_observation(self, msg:Float32MultiArray):
        if self.model is None:
            return

        l = int(len(msg.data) / 2)
        obs = np.array(msg.data, dtype=np.float32).reshape((l,2))
        self.model.fit(obs)

        hps = self._get_hmm_parameters()
        pub_msg = hmm_to_msg(*hps)
        self.pub_new_hmm.publish(pub_msg)

        self.get_logger().info("calced new hmm params")


    def cb_init_suggester(self, msg:Float32MultiArray):
        init_prob, tr_prob, avrs, covars, _, _ = msg_to_suggester(msg)
        self._set_hmm_parameters(init_prob, tr_prob, avrs, covars)
        self.get_logger().info("inited params")

        self.destroy_subscription(self.sub_init_suggester)
        self.destroy_timer(self.timer_req_init_suggester)


    def _get_hmm_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        covars_xx = self.model.covars_[:,0,0].astype(np.float32)
        covars_xy = self.model.covars_[:,0,1].astype(np.float32)
        covars_yy = self.model.covars_[:,1,1].astype(np.float32)
        covars = np.stack([covars_xx, covars_xy, covars_yy]).T

        return ( \
            self.model.startprob_.astype(np.float32), \
            self.model.transmat_.astype(np.float32), \
            self.model.means_.astype(np.float32), \
            covars \
        )


    def _set_hmm_parameters(self, init_prob:np.ndarray, tr_prob:np.ndarray, avrs:np.ndarray, covars:np.ndarray):
        n = init_prob.shape[0] # number of motions
        self.motion_num = n
        self.model = GaussianHMM(n_components=n, covariance_type='full')
        self.model.startprob_ = init_prob.astype(np.float32)
        self.model.transmat_ = tr_prob.astype(np.float32)
        self.model.means_ = avrs.astype(np.float32)
        covars_mat = np.ndarray((n,2,2), dtype=np.float32)
        covars_mat[:,0,0] = covars[:,0]
        covars_mat[:,0,1] = covars[:,1]
        covars_mat[:,1,0] = covars[:,1]
        covars_mat[:,1,1] = covars[:,2]
        self.model.covars_ = covars_mat


def main():
    rclpy.init()
    node = HMMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
