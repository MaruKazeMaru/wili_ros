# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

from hmmlearn.hmm import GaussianHMM
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from ._convert import hmm_parameter_to_ros_msg, ros_msg_to_hmm_parameter

class HMMUpdater(Node):
    def __init__(self):
        super().__init__('hmm_updater')

        self.motion_num:int = None
        self.model:GaussianHMM = None

        self.pub_new_hmm = self.create_publisher(Float32MultiArray, 'new_hmm', 1)
        self.sub_init_hmm = self.create_subscription(Float32MultiArray, "init_hmm", self.cb_init_hmm, 1)
        self.sub_obs = self.create_subscription(Float32MultiArray, "observation", self.cb_observation, 5)


    def cb_observation(self, msg:Float32MultiArray):
        if self.model is None:
            return

        l = int(len(msg.data) / 2)
        obs = np.array(msg.data, dtype=np.float32).reshape((l,2))
        self.model.fit(obs)

        hps = self._get_hmm_parameters()
        msg = hmm_parameter_to_ros_msg(*hps)
        self.pub_new_hmm.publish(msg)


    def cb_init_hmm(self, msg:Float32MultiArray):
        hps = ros_msg_to_hmm_parameter(msg)
        self._set_hmm_parameters(*hps)


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
    node = HMMUpdater()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
