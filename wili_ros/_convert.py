# SPDX-FileCopyrightText: 2024 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

def hmm_to_msg(
    init_prob:np.ndarray, tr_prob:np.ndarray, avrs:np.ndarray, covars:np.ndarray
) -> Float32MultiArray:
    n = init_prob.shape[0]
    dims = [
        MultiArrayDimension(label="motion_num", size=n),
    ]
    datas = [
        init_prob,
        tr_prob.flatten(),
        avrs.flatten(),
        covars.flatten(),
    ]
    msg = Float32MultiArray(
        layout=MultiArrayLayout(dim=dims, data_offset=0),
        data=np.concatenate(datas, axis=0)
    )
    return msg


def dens_to_msg(
    dens_miss_probs:np.ndarray
) -> Float32MultiArray:
    k = dens_miss_probs.shape[0]
    dim = MultiArrayDimension(label="sample_num", size=k),
    msg = Float32MultiArray(
        layout=MultiArrayLayout(dim=[dim], data_offset=0),
        data=dens_miss_probs
    )
    return msg


def suggester_to_msg(
    init_prob:np.ndarray, tr_prob:np.ndarray, avrs:np.ndarray, covars:np.ndarray, 
    miss_probs:np.ndarray, dens_miss_probs:np.ndarray
) -> Float32MultiArray:
    k, n = miss_probs.shape
    dims = [
        MultiArrayDimension(label="motion_num", size=n),
        MultiArrayDimension(label="sample_num", size=k),
    ]
    datas = [
        init_prob,
        tr_prob.flatten(),
        avrs.flatten(),
        covars.flatten(),
        miss_probs.flatten(),
        dens_miss_probs,
    ]
    msg = Float32MultiArray(
        layout=MultiArrayLayout(dim=dims, data_offset=0),
        data=np.concatenate(datas, axis=0)
    )
    return msg


def msg_to_hmm(
    msg:Float32MultiArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return (init_prob, tr_prob, avrs, covars)


def msg_to_dens(
    msg:Float32MultiArray
) -> np.ndarray:
    k:int = msg.layout.dim[0].size
    s = 0
    e = k
    dens_miss_probs = np.array(msg.data[s:e], dtype=np.float32)
    return dens_miss_probs


def msg_to_suggester(
    msg:Float32MultiArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n:int = msg.layout.dim[0].size
    k:int = msg.layout.dim[1].size
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
    s = e
    e += n * k
    miss_probs = np.array(msg.data[s:e], dtype=np.float32).reshape((k,n))
    s = e
    e += k
    dens_miss_probs = np.array(msg.data[s:e], dtype=np.float32)
    return (init_prob, tr_prob, avrs, covars, miss_probs, dens_miss_probs)
