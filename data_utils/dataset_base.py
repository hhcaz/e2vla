import os
import cv2
import time
import h5py
import torch
import random
import numpy as np
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional
from torchvision.transforms import v2
from torch.utils.data import (
    Dataset, IterableDataset, 
    ConcatDataset, ChainDataset, 
    get_worker_info, DataLoader
)

from . import h5io, align


def infer_record_dt(t: np.ndarray, default: float = 1.0):
    if len(t) < 2:
        return default
    else:
        return (t[-1] - t[0]) / (len(t) - 1)


def find_closest_ind(train: np.ndarray, query: np.ndarray):
    bin_indices = np.digitize(query, train)
    sample_indices = []
    
    # query < train
    mask = (bin_indices == 0)
    if np.any(mask):
        sample_indices.append(np.array([0]*mask.sum(), dtype=bin_indices.dtype))
    
    # query in train
    mask = (bin_indices > 0) & (bin_indices < len(train))
    r_ind = bin_indices[mask]
    l_ind = r_ind - 1
    
    dist0 = np.abs(train[l_ind] - query[mask])
    dist1 = np.abs(train[r_ind] - query[mask])
    sample_indices.append(np.where(dist0 < dist1, l_ind, r_ind))

    # query > train
    mask = (bin_indices == len(train))
    if np.any(mask):
        sample_indices.append(np.array([len(train)-1]*mask.sum(), dtype=bin_indices.dtype))
    
    sample_indices = np.concatenate(sample_indices)
    return sample_indices


class DataSampler(object):
    @classmethod
    def pad2ncam(self, x: np.ndarray, num_camera: int, dim: int, zero_init: bool):
        pad_ncam = num_camera - x.shape[dim]
        if pad_ncam < 0:
            raise ValueError("[ERR ] current ncam = {}, which is larger than desired ncam = {}"
                             .format(x.shape[dim], num_camera))
        if pad_ncam > 0:
            pad = x.take([-1]*pad_ncam, axis=dim)
            if zero_init:
                pad[:] = 0
            x = np.concatenate([x, pad], axis=dim)
        return x

    @classmethod
    def pad2nee(self, x: np.ndarray, num_ee: int, dim: int):
        pad_nee = num_ee - x.shape[dim]
        if pad_nee < 0:
            raise ValueError("[ERR ] current nee = {}, which is larger than desired nee = {}"
                             .format(x.shape[dim], num_ee))
        if pad_nee > 0:
            pad = x.take([-1]*pad_nee, axis=dim)
            x = np.concatenate([x, pad], axis=dim)
        return x
    
    @classmethod
    def preprocess_images(
        cls,
        Ks: List[np.ndarray],
        rgbs: List[np.ndarray],
        masks: Optional[List[np.ndarray]],
        output_image_hw: Optional[Tuple[int, int]] = None
    ):
        processed_Ks = []
        processed_rgbs = []
        processed_masks = []
        
        for cam_idx, rgb in enumerate(rgbs):
            # rgb: (T, 3, H, W)
            K = Ks[cam_idx]  # (3, 3)

            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.
            
            if masks is None or masks[cam_idx] is None:
                T, _, Hin, Win = rgb.shape
                mask = np.ones((T, Hin, Win), dtype=bool)
            else:
                mask = masks[cam_idx]
            
            if output_image_hw is not None:
                Hout, Wout = output_image_hw
                rgb, metadata = ImageProcessor.scale_to_fit(rgb, Hout, Wout)
                mask, metadata = ImageProcessor.scale_to_fit(mask, Hout, Wout)
                K = ImageProcessor.tform_K_for_scale_to_fit(K, **metadata)

                rgb, metadata = ImageProcessor.center_view(rgb, Hout, Wout)
                mask, metadata = ImageProcessor.center_view(mask, Hout, Wout)
                K = ImageProcessor.tform_K_for_center_view(K, **metadata)
            
            processed_Ks.append(K)
            processed_rgbs.append(rgb)
            processed_masks.append(mask)
        
        processed_Ks = np.stack(processed_Ks, axis=0)  # (Ncam, 3, 3)
        processed_rgbs = np.stack(processed_rgbs, axis=1)  # (T, Ncam, 3, H, W)
        processed_masks = np.stack(processed_masks, axis=1)  # (T, Ncam, H, W)
        return processed_Ks, processed_rgbs, processed_masks

    @classmethod
    def sample_framedict(
        cls,
        obs_traj: List[Dict[str, np.ndarray]], 
        ee_indices: List[int], 
        camera_names: List[str], 
        num_history_cameras: int, 
        num_history_states: int, 
        num_future_states: int, 
        latest: bool = False, 
        sample_state_gaps: int = 1,
        sample_camera_gaps: int = 1, 
        sample_dt: float = 1.0,
        record_dt: Optional[float] = None, 
        output_image_hw: Optional[Tuple[int, int]] = None, 
        enable_seg: bool = False, 
        pad2ncam: int = -1,
        pad2nee: int = -1
    ):
        """
        obs_traj is a list of dict containing necessary keys listed as followings.
        - ee_pose: np.ndarray of shape (4, 4) or (nee, 4, 4), ^{world}_{ee} T
        - gripper: float or np.ndarray of shape (nee,), value from [0 (close), 1 (open)]
        - timestamp: float, current timestamp
        - CAMERA_NAME_0: 
            - model: pinhole
            - camera:
                - width: int
                - height: int
                - K: np.ndarray of shape (3, 3) or (9,)
            - data:
                - color: np.ndarray, shape=(H, W, C)
                - seg: None | np.ndarray of shape (H, W) | isaacsim seg output
                - wcT: np.ndarray of shape (4, 4), ^{world}_{cam} T
        
        - CAMERA_NAME_1: similar as CAMERA_NAME_0
            - model: pinhole
            - ...
        """
        if not latest:
            last_obs_index = np.random.choice(len(obs_traj), 1)[0]
        else:
            last_obs_index = len(obs_traj) - 1

        all_timestamps = np.array([tau["timestamp"] for tau in obs_traj]).astype(np.float64)
        if record_dt is not None:
            # all_timestamps_calibrated = all_timestamps[-1] - np.arange(len(all_timestamps))[::-1] * record_dt
            all_timestamps_calibrated = all_timestamps
        else:
            all_timestamps_calibrated = all_timestamps
            record_dt = infer_record_dt(all_timestamps, default=sample_dt)
        
        if sample_dt is None:
            sample_dt = record_dt
        
        # we don't interpolate the images, but find the image with closest timestamp
        current_time = all_timestamps_calibrated[last_obs_index]
        prev_obs_sample_time = current_time + np.arange(-num_history_cameras+1, 1) * sample_camera_gaps * sample_dt
        prev_obs_sample_ind = find_closest_ind(all_timestamps_calibrated, prev_obs_sample_time)

        obs_across_cams: Dict[str, list] = {}
        for cam_name in camera_names:
            obs_cam = h5io.gather_frames(obs_traj, cam_name, prev_obs_sample_ind, compress=False)
            for k, v in obs_cam.items():
                if k not in obs_across_cams:
                    obs_across_cams[k] = []
                obs_across_cams[k].append(v)

        K, rgbs, masks = cls.preprocess_images(
            Ks=obs_across_cams["K"],
            rgbs=obs_across_cams["rgb"],
            masks=obs_across_cams.get("mask", None) if enable_seg else None,
            output_image_hw=output_image_hw
        )
        # K: (ncam, 3, 3); rgbs: (T, ncam, 3, H, W); masks: (T, ncam, H, W)
        
        cam_poses = np.stack(obs_across_cams["pose"], axis=1)   # (T, ncam, 4, 4)
        ee_poses = h5io.gather_ee_poses(obs_traj, prev_obs_sample_ind)  # (T, nee, 4, 4)
        
        # interpolate the robot states
        all_ee_poses = np.stack([tau["ee_pose"] for tau in obs_traj], axis=0)  # (L, nee, 4, 4)
        all_grippers = np.array([tau["gripper"] for tau in obs_traj])  # (L, nee)
        all_states = {"ee_pose": all_ee_poses, "gripper": all_grippers}
        interp_funcs = {"ee_pose": align.interp_SE3_sep, "gripper": align.interp_linear}
        
        history_time = current_time + np.arange(-num_history_states+1, 1) * sample_state_gaps * sample_dt
        future_time = current_time + np.arange(1, num_future_states+1) * sample_state_gaps * sample_dt

        history_queries = align.align_data(
            query_time=history_time,
            train_time=all_timestamps_calibrated,
            train_data=all_states,
            interp_funcs=interp_funcs
        )
        history_states = h5io.compose_ee_gripper(
            ee_poses=history_queries["ee_pose"], 
            grippers=history_queries["gripper"]
        )

        future_queries = align.align_data(
            query_time=future_time,
            train_time=all_timestamps_calibrated,
            train_data=all_states,
            interp_funcs=interp_funcs,
        )
        future_states = h5io.compose_ee_gripper(
            ee_poses=future_queries["ee_pose"],
            grippers=future_queries["gripper"]
        )

        # pad to n camera
        if pad2ncam > 0:
            rgbs = cls.pad2ncam(rgbs, pad2ncam, dim=1, zero_init=True)
            masks = cls.pad2ncam(masks, pad2ncam, dim=1, zero_init=True)
            cam_poses = cls.pad2ncam(cam_poses, pad2ncam, dim=1, zero_init=False)
            K = cls.pad2ncam(K, pad2ncam, dim=0, zero_init=False)
        
        # previous data has only one ee, therefore we preserve this for compatility
        if ee_poses.ndim == 3:
            ee_poses = ee_poses[:, None]  # (To, 4, 4) -> (To, Nee=1, 4, 4)
        if history_states.ndim == 2:
            history_states = history_states[:, None]  # (nhist, 17) -> (nhist, Nee=1, 17)
        if future_states.ndim == 2:
            future_states = future_states[:, None]  # (Ta, 17) -> (Ta, Nee=1, 17)
        
        # select ee by indices
        assert isinstance(ee_indices, (list, tuple))
        ee_poses = ee_poses.take(ee_indices, axis=1)
        history_states = history_states.take(ee_indices, axis=1)
        future_states = future_states.take(ee_indices, axis=1)
        
        # pad to n ee
        current_nee = len(ee_indices)
        if pad2nee > 0:
            ee_poses = cls.pad2nee(ee_poses, pad2nee, dim=1)
            history_states = cls.pad2nee(history_states, pad2nee, dim=1)
            future_states = cls.pad2nee(future_states, pad2nee, dim=1)
            valid_ee_mask = np.zeros(pad2nee, dtype=bool)
            valid_ee_mask[:current_nee] = True
        else:
            valid_ee_mask = np.ones(current_nee, dtype=bool)

        return (
            rgbs,                               # (To, ncam, 3, H, W)
            masks,                              # (To, ncam, H, W)
            cam_poses.astype(np.float32),       # (To, ncam, 4, 4)
            ee_poses.astype(np.float32),        # (To, nee, 4, 4)
            history_states.astype(np.float32),  # (nhist, nee, 17)
            future_states.astype(np.float32),   # (Ta, nee, 17)
            current_time,                       # scalar,
            K.astype(np.float32),               # (ncam, 3, 3)
            valid_ee_mask,                      # (nee,)
        )

    @classmethod
    def sample_hdf5(
        cls,
        obs_traj: h5py.File, 
        default_ee_indices: List[int], 
        camera_names: List[str], 
        num_history_cameras: int, 
        num_history_states: int, 
        num_future_states: int, 
        latest: bool = False, 
        sample_state_gaps: int = 1,
        sample_camera_gaps: int = 1,
        sample_dt: float = 1.0,
        record_dt: Optional[float] = None, 
        output_image_hw: Optional[Tuple[int, int]] = None, 
        enable_seg: bool = False, 
        pad2ncam: int = -1,
        pad2nee: int = -1, 
        video_root: Optional[str] = None,
        debug_sample_index: Optional[int] = None
    ):
        """
        obs_traj is a tree-like data structure
        - ee_pose: np.ndarray of shape (T, nee, 4, 4)
        - gripper: np.ndarray of shape (T, nee)
        - ee_pose_desired (optional): np.ndarray of shape (T, nee, 4, 4)
        - gripper_desired (optional): np.ndarray of shape (T, nee)
        - timestamp: np.ndarray of shape (T,)
        - CAMERA_NAME_0:
            - rgb: np.ndarray of shape (T, 3, H, W) or list of bytes (jpeg encoding)
            - pose: np.ndarray of shape (T, 4, 4)
            - K: np.ndarray of shape (3, 3), camera intrinsic
        - CAMERA_NAME_1:
            - rgb: np.ndarray of shape (T, 3, H, W) or list of vlen
            - ...
        """
        obs_traj_len = obs_traj["ee_pose"].len()
        if not latest:
            last_obs_index = np.random.choice(obs_traj_len, 1)[0]
        else:
            last_obs_index = obs_traj_len - 1
        
        if debug_sample_index is not None:
            print("[INFO] Debug sample index set, overwrite")
            last_obs_index = debug_sample_index

        all_timestamps = obs_traj["timestamp"][:].astype(np.float64)  # (L,)
        if record_dt is not None:
            all_timestamps_calibrated = np.arange(len(all_timestamps)) * record_dt
        else:
            all_timestamps_calibrated = all_timestamps
            record_dt = infer_record_dt(all_timestamps, default=sample_dt)
        
        if sample_dt is None:
            sample_dt = record_dt

        # we don't interpolate the images, but find the image with closest timestamp
        current_time = all_timestamps_calibrated[last_obs_index]
        prev_obs_sample_time = current_time + np.arange(-num_history_cameras+1, 1) * sample_camera_gaps * sample_dt
        prev_obs_sample_ind = find_closest_ind(all_timestamps_calibrated, prev_obs_sample_time)

        obs_across_cams: Dict[str, list] = {}
        for cam_name in camera_names:
            obs_cam = h5io.slice_encoded_frames(
                obs_traj[cam_name], 
                prev_obs_sample_ind,
                timestamp=all_timestamps,  # use original timestamp to iter video file
                video_root=video_root
            )
            for k, v in obs_cam.items():
                if k not in obs_across_cams:
                    obs_across_cams[k] = []
                obs_across_cams[k].append(v)
        
        K, rgbs, masks = cls.preprocess_images(
            Ks=obs_across_cams["K"],
            rgbs=obs_across_cams["rgb"],
            masks=obs_across_cams.get("mask", None) if enable_seg else None,
            output_image_hw=output_image_hw
        )
        # K: (ncam, 3, 3); rgbs: (T, ncam, 3, H, W); masks: (T, ncam, H, W)
        
        cam_poses = np.stack(obs_across_cams["pose"], axis=1)   # (T, ncam, 4, 4)
        ee_poses = h5io.slice_dset(obs_traj["ee_pose"], prev_obs_sample_ind)  # (T, 4, 4)

        # interpolate the robot states
        all_states = {
            "ee_pose": obs_traj["ee_pose"][:],  # (L, 4, 4)
            "gripper": obs_traj["gripper"][:],  # (L,)
        }

        interp_funcs = {
            "ee_pose": align.interp_SE3_sep, 
            "gripper": align.interp_linear
        }

        history_time = current_time + np.arange(-num_history_states+1, 1) * sample_state_gaps * sample_dt
        future_time = current_time + np.arange(1, num_future_states+1) * sample_state_gaps * sample_dt
        future_desired_time = current_time + np.arange(num_future_states) * sample_state_gaps * sample_dt
        # since desired has been given, this is one step behind future_time

        history_queries = align.align_data(
            query_time=history_time,
            train_time=all_timestamps_calibrated,
            train_data=all_states,
            interp_funcs=interp_funcs
        )
        history_states = h5io.compose_ee_gripper(
            ee_poses=history_queries["ee_pose"], 
            grippers=history_queries["gripper"]
        )
                
        if "gripper_desired" in obs_traj.keys():
            future_grippers = align.align_data(
                query_time=future_desired_time,
                train_time=all_timestamps_calibrated,
                train_data={"gripper": obs_traj["gripper_desired"][:]},
                interp_funcs={"gripper": align.interp_linear}
            )["gripper"]  # (L, nee)
        else:
            future_grippers = align.align_data(
                query_time=future_time,
                train_time=all_timestamps_calibrated,
                train_data={"gripper": all_states["gripper"]},
                interp_funcs={"gripper": align.interp_linear}
            )["gripper"]  # (L, nee)
        
        if "ee_pose_desired" in obs_traj.keys():
            future_ee_poses = align.align_data(
                query_time=future_desired_time,
                train_time=all_timestamps_calibrated,
                train_data={"ee_pose": obs_traj["ee_pose_desired"][:]},
                interp_funcs={"ee_pose": align.interp_SE3_sep}
            )["ee_pose"]  # (L, nee, 4, 4)
        else:
            future_ee_poses = align.align_data(
                query_time=future_time,
                train_time=all_timestamps_calibrated,
                train_data={"ee_pose": all_states["ee_pose"]},
                interp_funcs={"ee_pose": align.interp_SE3_sep}
            )["ee_pose"]  # (L, nee, 4, 4)
        
        future_states = h5io.compose_ee_gripper(
            ee_poses=future_ee_poses,
            grippers=future_grippers
        )  # (L, nee, 17)

        # pad to n camera
        if pad2ncam > 0:
            rgbs = cls.pad2ncam(rgbs, pad2ncam, dim=1, zero_init=True)
            masks = cls.pad2ncam(masks, pad2ncam, dim=1, zero_init=True)
            cam_poses = cls.pad2ncam(cam_poses, pad2ncam, dim=1, zero_init=False)
            K = cls.pad2ncam(K, pad2ncam, dim=0, zero_init=False)
        
        # previous data has only one ee, therefore we preserve this for compatility
        if ee_poses.ndim == 3:
            ee_poses = ee_poses[:, None]  # (To, 4, 4) -> (To, Nee=1, 4, 4)
        if history_states.ndim == 2:
            history_states = history_states[:, None]  # (nhist, 17) -> (nhist, Nee=1, 17)
        if future_states.ndim == 2:
            future_states = future_states[:, None]  # (Ta, 17) -> (Ta, Nee=1, 17)
        
        # select ee by indices
        if "ee_indices" in obs_traj.attrs:
            # this overwrite the config's ee_indices by h5 data's ee_indices
            # allow sample specific enabling/disabling which ee pose to predict
            ee_indices = obs_traj.attrs["ee_indices"]
            assert len(ee_indices) <= len(default_ee_indices), (
                "sample use ee_indices = {}, while default_ee_indices = {}".format(
                    ee_indices, default_ee_indices
                )
            )
        else:
            ee_indices = default_ee_indices
        
        assert isinstance(ee_indices, (list, tuple, np.ndarray))
        ee_poses = ee_poses.take(ee_indices, axis=1)
        history_states = history_states.take(ee_indices, axis=1)
        future_states = future_states.take(ee_indices, axis=1)
        
        # pad to n ee
        current_nee = len(ee_indices)
        if pad2nee > 0:
            ee_poses = cls.pad2nee(ee_poses, pad2nee, dim=1)
            history_states = cls.pad2nee(history_states, pad2nee, dim=1)
            future_states = cls.pad2nee(future_states, pad2nee, dim=1)
            valid_ee_mask = np.zeros(pad2nee, dtype=bool)
            valid_ee_mask[:current_nee] = True
        else:
            valid_ee_mask = np.ones(current_nee, dtype=bool)
        
        if obs_traj.attrs.get("is_bgr", False):
            # revert bgr to rgb
            rgbs = np.ascontiguousarray(np.flip(rgbs, axis=2))

        return (
            rgbs,                               # (To, ncam, 3, H, W)
            masks,                              # (To, ncam, H, W)
            cam_poses.astype(np.float32),       # (To, ncam, 4, 4)
            ee_poses.astype(np.float32),        # (To, nee, 4, 4)
            history_states.astype(np.float32),  # (nhist, nee, 17)
            future_states.astype(np.float32),   # (Ta, nee, 17)
            current_time,                       # scalar,
            K.astype(np.float32),               # (ncam, 3, 3)
            valid_ee_mask,                      # (nee,)
        )


def gen_norm_xy_map(H: int, W: int, K: np.ndarray):
    """
    Args:
        H (int): image height
        W (int): image width
        K (np.ndarray): (Ncam, 3, 3)
    
    Returns:
        norm_xy (np.ndarray): (Ncam, 2, H, W)
    """
    fx = K[:, 0, 0]; fy = K[:, 1, 1]; cx = K[:, 0, 2]; cy = K[:, 1, 2]  # (ncam,)
    XX, YY = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    grid = np.stack([XX, YY], axis=0)  # (2, H, W)
    cxy = np.stack([cx, cy], axis=-1)  # (ncam, 2)
    fxy = np.stack([fx, fy], axis=-1)  # (ncam, 2)
    norm_xy = (grid - cxy[:, :, None, None]) / fxy[:, :, None, None]  # (ncam, 2, H, W)
    return norm_xy


@dataclass
class DataConfig(object):
    ### total traj time of gt future action is `sample_dt * num_future_states`
    sample_dt: float
    ### if None, inferenced from `timestamp` key from data, otherwise overwrite the data
    record_dt: Optional[float]
    ### image height and width, none means remain unchanged
    output_image_hw: Optional[Tuple[int, int]]

    ### used in training and real-world execution
    ee_indices: Tuple[int]
    camera_names: Tuple[str]
    enable_seg: bool = False  # segment image patches if mask is available

    sample_state_gaps: int = 1
    sample_camera_gaps: int = 4

    num_history_cameras: int = 1
    num_history_states: int = 1
    num_future_states: int = 32  # future states as gt action

    video_root: Optional[str] = None  # for those rgb key as string, which means load video from video_root/rgb
    shuffle_cameras: bool = True


class ImageProcessor(object):
    @classmethod
    def scale_to_fit(cls, x: np.ndarray, H, W):
        old_H, old_W = x.shape[-2:]
        scale_H = H / old_H
        scale_W = W / old_W
        scale = min(scale_H, scale_W)
        
        new_H = int(scale * old_H)
        new_W = int(scale * old_W)
        scale_H = new_H / old_H
        scale_W = new_W / old_W
        
        x: Tensor = torch.from_numpy(x)
        if x.is_floating_point():
            x_resize = v2.Resize((new_H, new_W), v2.InterpolationMode.BILINEAR)(x)
        else:
            x_resize = v2.Resize((new_H, new_W), v2.InterpolationMode.NEAREST)(x)
        x_resize: np.ndarray = x_resize.numpy()
        
        metadata = dict(
            old_H=old_H,
            old_W=old_W,
            new_H=new_H, 
            new_W=new_W,
        )
        
        return x_resize, metadata

    @classmethod
    def tform_K_for_scale_to_fit(
        cls, 
        K: np.ndarray, 
        old_H: int, old_W: int, new_H: int, new_W: int
    ):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        
        scale_x = new_W / old_W
        scale_y = new_H / old_H
        
        new_fx = fx * scale_x
        new_fy = fy * scale_y
        new_cx = cx * scale_x
        new_cy = cy * scale_y
        
        K_new = K.copy()
        K_new[..., 0, 0] = new_fx
        K_new[..., 1, 1] = new_fy
        K_new[..., 0, 2] = new_cx
        K_new[..., 1, 2] = new_cy
        return K_new

    @classmethod
    def center_view(cls, x: np.ndarray, H, W):
        old_H, old_W = x.shape[-2:]
        dx = (W - old_W) // 2
        dy = (H - old_H) // 2
        metadata = dict(dcx=dx, dcy=dy)
        
        if old_H == H and old_W == W:
            return x, metadata
        
        x: Tensor = torch.from_numpy(x)
        x_view: np.ndarray = v2.CenterCrop((H, W))(x).numpy()
        return x_view, metadata
    
    @classmethod
    def tform_K_for_center_view(cls, K: np.ndarray, dcx: int, dcy: int):
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        K_new = K.copy()
        K_new[..., 0, 2] = cx + dcx
        K_new[..., 1, 2] = cy + dcy
        return K_new


class H5DatasetMapBase(Dataset):

    config = DataConfig(
        sample_dt=1.0,
        record_dt=None,
        output_image_hw=None, 
        ee_indices=(),
        camera_names=(),
    )

    def __init__(
        self, 
        h5_filelist: List[str], 
    ):
        self.h5_filelist = h5_filelist
        self.data_sampler = DataSampler()

        if isinstance(self.config.camera_names, str):
            # wrap to tuple
            self.config.camera_names = (self.config.camera_names,)
        
        if isinstance(self.config.ee_indices, int):
            # wrap to tuple
            self.config.ee_indices = (self.config.ee_indices,)
        
        self.cam_num = len(self.config.camera_names)
        self.ee_num = len(self.config.ee_indices)
        self.pad2ncam = self.cam_num
        self.pad2nee = self.ee_num
    
    @classmethod
    def inst(cls) -> "H5DatasetMapBase":
        raise NotImplementedError

    def __len__(self):
        return len(self.h5_filelist)
    
    def sample_from_hdf5(
        self, 
        h5: h5py.File, 
        latest: bool = False, 
        debug_sample_index: Optional[int] = None
    ):
        prompt_candidates = []
        for k, v in h5.attrs.items():
            if "prompt_text" in k:
                v = v.strip()
                if len(v):
                    prompt_candidates.append(v)
        
        prompt_text = random.sample(prompt_candidates, 1)[0] if len(prompt_candidates) else ""
        if len(prompt_text) == 0:
            prompt_text = "Do any possible actions"

        if self.config.shuffle_cameras:
            camera_names = list(self.config.camera_names).copy()
            random.shuffle(camera_names)
        else:
            camera_names = self.config.camera_names

        (
            obs_rgbs, obs_masks, obs_cam_poses, obs_ee_poses, 
            history_states, future_states, timestamps, K, valid_ee_mask
        ) = self.data_sampler.sample_hdf5(
            obs_traj=h5, 
            default_ee_indices=self.config.ee_indices,
            camera_names=camera_names, 
            num_history_cameras=self.config.num_history_cameras, 
            num_history_states=self.config.num_history_states, 
            num_future_states=self.config.num_future_states,
            latest=latest, 
            sample_state_gaps=self.config.sample_state_gaps, 
            sample_camera_gaps=self.config.sample_camera_gaps, 
            sample_dt=self.config.sample_dt,
            record_dt=self.config.record_dt, 
            output_image_hw=self.config.output_image_hw,
            enable_seg=self.config.enable_seg,
            pad2ncam=self.pad2ncam,
            pad2nee=self.pad2nee,
            video_root=self.config.video_root,
            debug_sample_index=debug_sample_index
        )

        T, ncam, C, H, W = obs_rgbs.shape
        norm_xys = gen_norm_xy_map(H, W, K).astype(np.float32)
        norm_xys = norm_xys[None].repeat(T, axis=0)  # (T, ncam, 2, H, W)
        
        out = {
            "K": K,                                 # (ncam, 3, 3)
            "obs_rgbs": obs_rgbs,                   # (To, ncam, 3, H, W)
            "obs_masks": obs_masks,                 # (To, ncam, H, W)
            "prompt_text": prompt_text,             # str
            "obs_norm_xys": norm_xys,               # (To, ncam, 2, H, W)
            "obs_extrinsics": obs_cam_poses,        # (To, ncam, 4, 4)
            "current_ee_pose": obs_ee_poses[-1],    # (nee, 4, 4)
            "history_ee_states": history_states,    # (nhist, nee, 17)
            "gt_future_ee_states": future_states,   # (Ta, nee, 17)
            "timestamps": timestamps,               # (To,)
            "valid_ee_mask": valid_ee_mask,         # (nee,)
        }
        return out

    def __getitem__(self, i):
        h5_file = self.h5_filelist[i]

        with h5py.File(h5_file, "r") as h5:
            out = self.sample_from_hdf5(h5, latest=False, debug_sample_index=None)
        return out
    
    def visualize(self):
        return visualize_dataset(self)


class H5DatasetIterBase(H5DatasetMapBase, IterableDataset):
    
    def __init__(self, h5_filelist: List[str]):
        super().__init__(h5_filelist)
        self._shuffle_h5_list = False
    
    def __iter__(self):
        indices = np.arange(len(self.h5_filelist))
        if self._shuffle_h5_list:
            np.random.shuffle(indices)
            # print("[INFO] {} shuffles dataset".format(os.getpid()))
        
        # worker_info = get_worker_info()
        # if (worker_info is not None) and (worker_info.num_workers > 1):
        #     # split workload
        #     splits = np.linspace(0, len(indices), num=worker_info.num_workers+1, endpoint=True)
        #     splits = splits.astype(np.int64).tolist()
        #     indices = indices[splits[worker_info.id]:splits[worker_info.id+1]].copy()
        
        for i in indices:
            # print(os.getpid())
            yield self[int(i)]


def concat_datasets(
    datasets: List[Union[H5DatasetMapBase, H5DatasetIterBase]],
    shuffle: bool = None
):
    num_cams = [d.cam_num for d in datasets]
    num_ees = [d.ee_num for d in datasets]
    pad2ncam = max(num_cams)
    pad2nee = max(num_ees)
    for d in datasets:
        d.pad2ncam = pad2ncam
        d.pad2nee = pad2nee
        print("[INFO] dataset {} uses {} cameras, {} end-effectors".format(d, d.cam_num, d.ee_num))
    print("[INFO] Final padded camera num: {}, end-effector num: {}".format(pad2ncam, pad2nee))
    
    if isinstance(datasets[0], H5DatasetIterBase):
        if shuffle:
            for d in datasets:
                d._shuffle_h5_list = True
        return ChainDataset(datasets)
    else:
        return ConcatDataset(datasets)


def get_dataloader(
    datasets: List[Union[H5DatasetMapBase, H5DatasetIterBase]],
    batch_size: int,
    num_workers: int = 0,
    shuffle: Optional[bool] = None, 
    persistent_workers: bool = False,
    sample_weights: Optional[list] = None,
    sample_multiplex: int = 1
):
    if sample_weights is not None:
        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights / sample_weights.sum()
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights.tolist(), 
            num_samples=len(sample_weights) * sample_multiplex, 
            replacement=True
        )
    else:
        sampler = None
    
    if isinstance(datasets, (list, tuple)):
        datasets = concat_datasets(datasets, shuffle)
    elif isinstance(datasets, H5DatasetIterBase):
        if shuffle == True:
            datasets._shuffle_h5_list = True
    
    if isinstance(datasets, (H5DatasetIterBase, ChainDataset)):
        shuffle = None  # overwrite shuffle args
    
    dataloader = DataLoader(
        dataset=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        shuffle=shuffle, 
        sampler=sampler,
    )
    return dataloader


def generate_sample_weights(
    datasets: List[H5DatasetMapBase],
    dataset_weights: List[float]
):
    sample_weights = []
    for i, dataset in enumerate(datasets):
        sample_weights.append(
            np.array([dataset_weights[i] / len(dataset)] * len(dataset))
        )
    sample_weights = np.concatenate(sample_weights)
    sample_weights = sample_weights / sample_weights.sum()
    return sample_weights.tolist()


def rbd(d: Dict[str, Tensor]):
    """remove batch dimension"""
    return {k:v[0] if v is not None else v for k, v in d.items()}


def proj(K: np.ndarray, cwT: np.ndarray, pos: np.ndarray):
    pos_in_cam = pos @ cwT[:3, :3].T + cwT[:3, 3]
    xy = pos_in_cam[:2] / pos_in_cam[-1:]

    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    fxy = np.array([fx, fy]); cxy = np.array([cx, cy])
    uv = xy * fxy + cxy
    return uv


def draw_ee_proj(bgr: np.ndarray, K: np.ndarray, cwT: np.ndarray, pose: np.ndarray):
    alen = 0.05  # 5cm
    pos = pose[:3, 3]
    x_end = tuple(proj(K, cwT, pos + pose[:3, 0] * alen).astype(int).tolist())
    y_end = tuple(proj(K, cwT, pos + pose[:3, 1] * alen).astype(int).tolist())
    z_end = tuple(proj(K, cwT, pos + pose[:3, 2] * alen).astype(int).tolist())
    origin = tuple(proj(K, cwT, pos).astype(int).tolist())
    
    cv2.line(bgr, origin, x_end, (0, 0, 255), thickness=2)
    cv2.line(bgr, origin, y_end, (0, 255, 0), thickness=2)
    cv2.line(bgr, origin, z_end, (255, 0, 0), thickness=2)
    return bgr


def visualize_traj(data: Dict[str, Tensor]):
    # data["obs_rgbs"]: (To, ncam, C, H, W)
    To, ncam, _, H, W = data["obs_rgbs"].shape
    rgb = rearrange(data["obs_rgbs"][-1], "n c h w -> n h w c")  # latest time
    
    # data["K"]: (ncam, 3, 3)
    K = data["K"]  # (ncam, 3, 3)
    
    # data["obs_extrinsics"]: (To, ncam, 4, 4)
    wcT = data["obs_extrinsics"][-1]  # (ncam, 4, 4)
    
    # data["gt_future_ee_states"]: (Ta, nee, 4*4+1)
    Ta, nee, _ = data["gt_future_ee_states"].shape
    weTs = data["gt_future_ee_states"][:, :, :16].view(Ta, nee, 4, 4)  # (Ta, nee, 4, 4)

    nhist, nee, _ = data["history_ee_states"].shape
    history_weTs = data["history_ee_states"][:, :, :16].view(nhist, nee, 4, 4)

    ceTs = (
        rearrange(torch.inverse(wcT), "ncam r c -> () () ncam r c") @ 
        rearrange(weTs, "Ta nee r c -> Ta nee () r c")
    )  # (Ta, nee, ncam, 4, 4)
    cets = ceTs[..., :3, 3]  # (Ta, nee, ncam, 3)

    proj_norm = cets[..., :2] / cets[..., 2:3]  # (Ta, nee, ncam, 2)
    fxy = K[..., [0, 1], [0, 1]]; cxy = K[..., [0, 1], [2, 2]]  # (ncam, 2)
    proj_pix = proj_norm * fxy + cxy  # (Ta, nee, ncam, 2)

    bgrs = np.ascontiguousarray(rgb.flip(-1).cpu().numpy())  # (ncam, H, W, C)
    proj_pix: np.ndarray = proj_pix.cpu().numpy()  # (Ta, nee, ncam, 2)
    
    # to cpu and numpy
    K = K.cpu().numpy()
    wcT = wcT.cpu().numpy()
    history_weTs = history_weTs.cpu().numpy()
    
    for eei in range(nee):        
        for cami in range(ncam):
            for x, y in proj_pix[:, eei, cami]:
                cv2.circle(bgrs[cami], (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)
    
            draw_ee_proj(
                bgrs[cami], 
                K=K[cami], 
                cwT=np.linalg.inv(wcT[cami]), 
                pose=history_weTs[-1, eei]
            )

    bgrs = rearrange(bgrs, "n h w c -> h (n w) c")
    bgrs = np.ascontiguousarray(bgrs)
    return bgrs

    # cv2.imshow("traj", bgrs)
    # key = cv2.waitKey(0)
    # return key


def visualize_dataset(
    dataset: H5DatasetMapBase
):
    episode_idx = 0
    frame_idx = 0

    original_shuffle_config = dataset.config.shuffle_cameras
    if dataset.config.shuffle_cameras:
        dataset.config.shuffle_cameras = False
        print("[INFO] In dataset visualization, temporarily set shuffle_cameras to False!!!")

    h5 = h5py.File(dataset.h5_filelist[episode_idx], mode="r")
    max_frames = len(h5["ee_pose"][:])

    # print("="*61)
    # print("[INFO] Usage:")
    # print("p: previous episode")
    # print("n: next episode")
    # print("<-: previous frame")
    # print("->: next frame")
    # print("="*61)
    # print()
    
    while True:
        print("-"*61)
        print("[INFO] Usage:")
        print("q: quit")
        print("p: previous episode")
        print("n: next episode")
        print("<-: previous frame")
        print("->: next frame")
        print()

        print("[INFO] Episode: {}/{}, frame: {}/{}".format(
            episode_idx+1, len(dataset), frame_idx, max_frames))
        
        out = dataset.sample_from_hdf5(
            h5=h5,
            latest=False,
            debug_sample_index=frame_idx
        )
        
        print("[INFO] Prompt text: {}".format(out["prompt_text"]))
        
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)
        
        bgrs = visualize_traj(
            data=out,
            # future_ee_states=[out["gt_future_ee_states"]],
            # colors=[(0, 255, 0)]
        )
        
        print("[INFO] future grippers = \n{}".format(
            out["gt_future_ee_states"][:, out["valid_ee_mask"], -1].transpose(0, 1)))
        
        cv2.imshow("gt traj", bgrs)
        key = cv2.waitKey(0)
        
        if key == ord("n"):
            h5.close()
            episode_idx = (episode_idx + 1) % len(dataset.h5_filelist)
            frame_idx = 0
            h5 = h5py.File(dataset.h5_filelist[episode_idx], mode="r")
            max_frames = len(h5["ee_pose"][:])
        elif key == ord('p'):
            h5.close()
            episode_idx = (episode_idx - 1) % len(dataset.h5_filelist)
            frame_idx = 0
            h5 = h5py.File(dataset.h5_filelist[episode_idx], mode="r")
            max_frames = len(h5["ee_pose"][:])
        elif key == 83:
            frame_idx = (frame_idx + 1) % max_frames
        elif key == 81:
            frame_idx = (frame_idx - 1) % max_frames
        elif key == ord('q'):
            h5.close()
            break
    
    dataset.config.shuffle_cameras = original_shuffle_config
    return

