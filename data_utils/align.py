import torch
import numpy as np
import pypose as pp
from typing import Dict, Callable


def interp_linear(q0: np.ndarray, q1: np.ndarray, t0, t1, t) -> np.ndarray:
    """
    Args:
        q0 (np.ndarray): (T, ...)
        q1 (np.ndarray): (T, ...)
        t0 (np.ndarray or float): (T,) or scalar
        t0 (np.ndarray or float): (T,) or scalar
        t (np.ndarray): (T,)
    
    Returns:
        q (np.ndarray): (T, ...)
    """
    delta = q1 - q0
    alpha = (t - t0) / (t1 - t0)
    if isinstance(alpha, np.ndarray) and isinstance(delta, np.ndarray):
        if alpha.ndim < delta.ndim:
            alpha.shape = alpha.shape + (1,) * (delta.ndim - alpha.ndim)
    update = alpha * delta
    q = q0 + update.astype(q0.dtype)
    return q


def interp_SO3(R0: np.ndarray, R1: np.ndarray, t0, t1, t) -> np.ndarray:
    """
    Args:
        R0 (np.ndarray): (T, ..., 3, 3)
        R1 (np.ndarray): (T, ..., 3, 3)
        t0 (np.ndarray or float): (T,) or scalar
        t0 (np.ndarray or float): (T,) or scalar
        t (np.ndarray): (T,)
    
    Returns:
        R (np.ndarray): (T, ..., 3, 3)
    """
    Log = pp.SO3_type.Log
    Exp = pp.so3_type.Exp
    Mul = pp.SO3_type.Mul
    Inv = pp.SO3_type.Inv
    
    R0 = pp.from_matrix(torch.from_numpy(R0), pp.SO3_type)
    R1 = pp.from_matrix(torch.from_numpy(R1), pp.SO3_type)
    delta = Log(Mul(Inv(R0), R1)).tensor()
    alpha = (t - t0) / (t1 - t0)
    
    if isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha).to(delta)
        if alpha.ndim < delta.ndim:
            alpha = alpha.view(*alpha.shape, *([1] * (delta.ndim - alpha.ndim)))
    
    update = alpha * delta
    R = Mul(R0, Exp(update)).matrix()
    return R.numpy()


def interp_SE3(T0: np.ndarray, T1: np.ndarray, t0, t1, t) -> np.ndarray:
    """
    Args:
        T0 (np.ndarray): (T, ..., 4, 4)
        T1 (np.ndarray): (T, ..., 4, 4)
        t0 (np.ndarray or float): (T,) or scalar
        t0 (np.ndarray or float): (T,) or scalar
        t (np.ndarray): (T,)
    
    Returns:
        T (np.ndarray): (T, ..., 4, 4)
    """
    Log = pp.SE3_type.Log
    Exp = pp.se3_type.Exp
    Mul = pp.SE3_type.Mul
    Inv = pp.SE3_type.Inv
    
    T0 = pp.from_matrix(torch.from_numpy(T0), pp.SE3_type)
    T1 = pp.from_matrix(torch.from_numpy(T1), pp.SE3_type)
    delta = Log(Mul(Inv(T0), T1)).tensor()
    alpha = (t - t0) / (t1 - t0)
    
    if isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        if alpha.ndim < delta.ndim:
            alpha = alpha.view(*alpha.shape, *([1] * (delta.ndim - alpha.ndim)))
    
    update = alpha * delta
    T = Mul(T0, Exp(update)).matrix()
    return T.numpy().astype(T0.dtype)


def interp_SE3_sep(T0: np.ndarray, T1: np.ndarray, t0, t1, t) -> np.ndarray:
    """
    Args:
        T0 (np.ndarray): (T, ..., 4, 4)
        T1 (np.ndarray): (T, ..., 4, 4)
        t0 (np.ndarray or float): (T,) or scalar
        t0 (np.ndarray or float): (T,) or scalar
        t (np.ndarray): (T,)
    
    Returns:
        T (np.ndarray): (T, ..., 4, 4)
    """
    R = interp_SO3(T0[..., :3, :3], T1[..., :3, :3], t0, t1, t)
    trans = interp_linear(T0[..., :3, 3], T1[..., :3, 3], t0, t1, t)
    shape = R.shape[:-2] + (4, 4)
    T = np.zeros(shape, R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = trans
    T[..., 3, 3] = 1
    return T


def align_data(
    query_time: np.ndarray, 
    train_time: np.ndarray,
    train_data: Dict[str, np.ndarray],
    interp_funcs: Dict[str, Callable]
) -> Dict[str, np.ndarray]:
    
    query_data = {k: [] for k in train_data}
    bin_indices = np.digitize(query_time, train_time)

    # query_time < min(train_time)
    mask = (bin_indices == 0)
    if np.any(mask):
        for k in query_data:
            query_data[k].append(np.repeat(train_data[k][0:1], repeats=mask.sum(), axis=0))
    
    # query_time in train_time
    mask = (bin_indices > 0) & (bin_indices < len(train_time))
    if np.any(mask):
        l_ind = bin_indices[mask] - 1
        t0 = train_time[l_ind]
        t1 = train_time[l_ind + 1]
        for k, f in interp_funcs.items():
            q0 = train_data[k][l_ind]
            q1 = train_data[k][l_ind + 1]
            query_data[k].append(f(q0, q1, t0, t1, query_time[mask]))
        
    # query_time > max(train_time)
    mask = (bin_indices == len(train_time))
    if np.any(mask):
        for k in query_data:
            query_data[k].append(np.repeat(train_data[k][-1:], repeats=mask.sum(), axis=0))
    
    for k in query_data:
        query_data[k] = np.concatenate(query_data[k], axis=0)
    return query_data
    
