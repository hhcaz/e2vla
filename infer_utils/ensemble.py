import torch
import numpy as np
import pypose as pp
from collections import deque
from data_utils.align import interp_linear, interp_SO3


class TrajEnsembler(object):
    def __init__(self, max_hist_len=-1):
        maxlen = max_hist_len if max_hist_len > 0 else None
        self.history_traj = deque(maxlen=maxlen)
        self.history_time = deque(maxlen=maxlen)
    
    def reset(self):
        self.history_time.clear()
        self.history_time.clear()
    
    def _weighted_sum_linear(cls, trajs: np.ndarray, weights: np.ndarray):
        weights.shape = weights.shape + (1,) * (trajs.ndim - weights.ndim)  # (N, T, 1)
        ensembled_traj = (trajs * weights).sum(axis=0) / weights.sum(axis=0)
        return ensembled_traj
    
    def _weighted_sum_SO3(cls, trajs: np.ndarray, weights: np.ndarray):
        """
        Args:
            trajs (np.ndarray): (N, T, 3, 3)
            weights (np.ndarray): (N, T)
        
        Returns:
            ensembled_traj (np.ndarray): (T, 3, 3)
        """
        Log = pp.SO3_type.Log
        Exp = pp.so3_type.Exp
        Mul = pp.SO3_type.Mul
        Inv = pp.SO3_type.Inv
        
        trajs = pp.from_matrix(torch.from_numpy(trajs), pp.SO3_type)
        init_SO3 = trajs[-1]  # (T, 3, 3)
        delta_so3 = Log(Mul(Inv(init_SO3), trajs)).tensor()  # (N, T, 3)
        
        weights.shape = weights.shape + (1,) * (delta_so3.ndim - weights.ndim)  # (N, T, 1)
        weights = torch.from_numpy(weights).to(delta_so3)
        update_so3 = (delta_so3 * weights).sum(dim=0) / weights.sum(dim=0)
        
        ensembled_traj = Mul(init_SO3, Exp(update_so3)).matrix()
        return ensembled_traj.numpy()
    
    def update(self, new_traj: np.ndarray, new_time: np.ndarray, on_SO3: bool = False):
        """
        Args:
            new_traj (np.ndarray): (T, D) or (T, D1, D2)
            new_time (np.ndarray): (T,), T is the length of predicted trajectory
            on_SO3 (bool): update in linear space or SO3 space
        
        Returns:
            ensembled_traj (np.ndarray): same shape as new_traj
        """
        self.history_traj.append(new_traj)  # (nhist, T, ...)
        self.history_time.append(new_time)  # (nhist, T)

        while True:
            initial_end_time = self.history_time[0][-1]
            current_start_time = self.history_time[-1][0]
            if current_start_time > initial_end_time:
                self.history_traj.popleft()
                self.history_time.popleft()
            else:
                break

        candidate_trajs = []
        candidate_weights = []
        if len(self.history_traj) > 1:
            for prev_traj, prev_time in zip(list(self.history_traj)[:-1], 
                                            list(self.history_time)[:-1]):
                bin_indices = np.digitize(new_time, prev_time)
                mask = (bin_indices > 0) & (bin_indices < len(prev_time))
                candidate_weights.append(mask.astype(np.float32))
                
                candidate_traj = new_traj.copy()
                l_ind = bin_indices[mask] - 1
                t0 = prev_time[l_ind]
                t1 = prev_time[l_ind + 1]
                
                q0 = prev_traj[l_ind]
                q1 = prev_traj[l_ind + 1]
                if mask.any():
                    if on_SO3:
                        candidate_traj[mask] = interp_SO3(q0, q1, t0, t1, new_time[mask])
                    else:
                        candidate_traj[mask] = interp_linear(q0, q1, t0, t1, new_time[mask])
                candidate_trajs.append(candidate_traj)
        
        candidate_trajs.append(new_traj)
        candidate_weights.append(np.ones(len(new_traj)))
        
        candidate_trajs = np.asarray(candidate_trajs)  # (N, T, D) or (N, T, D1, D2)
        candidate_weights = np.asarray(candidate_weights)  # (N, T)
        
        # set decay factor
        decay_factor = self.ensemble_weights(candidate_weights.shape[0])  # (N,)
        candidate_weights = candidate_weights * decay_factor[:, None]
        
        if on_SO3:
            ensembled_traj = self._weighted_sum_SO3(candidate_trajs, candidate_weights)
        else:
            ensembled_traj = self._weighted_sum_linear(candidate_trajs, candidate_weights)
        return ensembled_traj
    
    @property
    def num_history(self):
        return len(self.history_time)

    def ensemble_weights(self, N: int):
        """
        Args:
            N (int): number of history trajectory
        
        Returns:
            weight (np.ndarray): shape = (N,), value from small (earliest) to large (latest)
        """
        return np.linspace(0.1, 1, N)

