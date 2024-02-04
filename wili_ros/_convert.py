# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

def hmm_parameter_to_ros_msg(init_prob:np.ndarray, tr_prob:np.ndarray, avrs:np.ndarray, covars:np.ndarray) -> Float32MultiArray:
    n = init_prob.shape[0]
    dim_s = MultiArrayDimension(label="init_prob", size=n)
    dim_A = MultiArrayDimension(label="tr_prob", size=n*n)
    dim_mu = MultiArrayDimension(label="avrs", size=n*2)
    dim_Sigma = MultiArrayDimension(label="covars", size=n*3)
    datas = [ \
        init_prob.flatten(), \
        tr_prob.flatten(), \
        avrs.flatten(), \
        covars.flatten() \
    ]
    msg = Float32MultiArray( \
        layout=MultiArrayLayout(dim=[dim_s, dim_A, dim_mu, dim_Sigma], data_offset=0), \
        data=np.concatenate(datas, axis=0) \
    )
    return msg


def ros_msg_to_hmm_parameter(msg:Float32MultiArray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n:int = msg.layout.dim[0].size
    s = 0
    e = n
    init_prob = np.array(msg.data[s:e], dtype=np.float32)
    s = e
    e += n * n
    tr_prob = np.array(msg.data[s:e], dtype=np.float32).reshape((n,n))
    s = e
    e += n * 2
    avrs = np.array(msg.data[s:e], dtype=np.float32).reshape((n,2))
    s = e
    e += n * 3
    covars = np.array(msg.data[s:e], dtype=np.float32).reshape((n,3))
    return ( \
        init_prob, \
        tr_prob, \
        avrs, \
        covars \
    )
