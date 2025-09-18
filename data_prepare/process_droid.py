import os
import h5py
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, load_dataset



parser = argparse.ArgumentParser()
parser.add_argument("--input_root", type=str, default="./data_raw/droid_1.0.1")
parser.add_argument("--alter_vid_root", type=str, default="")
parser.add_argument("--output_root", type=str, default="./data_converted/droid")
parser.add_argument("--skip_saved", action="store_true", default=False)
opt = parser.parse_args()


print("Loading metadata...", flush=True)
dataset_root = Path(opt.input_root)
ds_meta = LeRobotDatasetMetadata(
    repo_id="cadene/droid_1.0.1",
    root=dataset_root,
    force_cache_sync=False
)



print(f"Total number of episodes: {ds_meta.total_episodes}", flush=True)
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}", flush=True)
print(f"Frames per second used during data collection: {ds_meta.fps}", flush=True)
print(f"Robot type: {ds_meta.robot_type}", flush=True)
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n", flush=True)
print(ds_meta)



def xyzrpy2mat(xyzrpy: np.ndarray):
    pose = np.eye(4)
    if xyzrpy.ndim > 1:
        B = xyzrpy.shape[0]
        pose = pose[None].repeat(B, 0)
    
    pose[..., :3, :3] = Rotation.from_euler("xyz", xyzrpy[..., 3:]).as_matrix()
    pose[..., :3, 3] = xyzrpy[..., :3]
    return pose


def save_to_h5(path: str, data_dict: dict, attr_dict: dict):
    """
    data_dict:
    - ee_pose: np.ndarray of shape (T, 4, 4)
    - gripper: np.ndarray of shape (T,)
    - CAMERA_NAME_0:
        - rgb: np.ndarray of shape (T, 3, H, W) or list of vlen
        - pose: np.ndarray of shape (4, 4)
        - K: np.ndarray of shape (3, 3), camera intrinsic
        - time: np.ndarray of shape (T,)
    - CAMERA_NAME_1:
        - rgb: np.ndarray of shape (T, 3, H, W) or list of vlen
        - ...
    
    attr_dict:
    - prompt_text: str
    - compress: bool
    """
    output_dir = os.path.dirname(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(path, "w") as h5:
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                h5.create_dataset(k, data=v)
            else:
                if isinstance(v[0], np.ndarray):
                    dtype = h5py.vlen_dtype(v[0].dtype)
                elif isinstance(v[0], str):
                    dtype = h5py.string_dtype(encoding='utf-8')
                else:
                    raise TypeError("Unknown type: {}".format(type(v[0])))
                
                dset = h5.create_dataset(k, shape=(len(v),), dtype=dtype)
                for i, vi in enumerate(v):
                    dset[i] = vi
        
        for k, v in attr_dict.items():
            h5.attrs[k] = v


h5_droid_root = Path(opt.output_root)
os.makedirs(str(h5_droid_root), exist_ok=True)


default_K = np.array([
    [522.42260742/4,   0.        , 653.9631958 /4],
    [  0.        , 522.42260742/4, 358.79196167/4],
    [  0.        ,   0.        ,   1.        ],
])


# video_root = Path(opt.)

if len(opt.alter_vid_root) > 0:
    video_root = Path(opt.alter_vid_root)
else:
    video_root = Path(opt.input_root)


for ep_id in tqdm(range(ds_meta.total_episodes)):

    print("-"*61)
    t0 = time.time()
    parquet_file = ds_meta.get_data_file_path(ep_id)
    hf_dataset = load_dataset("parquet", data_files=str(ds_meta.root / parquet_file), split="train")
    hf_dataset.set_transform(hf_transform_to_torch)

    success = hf_dataset["is_episode_successful"][0]
    if not success:
        print("[INFO] episode {} faild, skip".format(parquet_file))
        continue
    
    ep_idx = hf_dataset["episode_index"][0].item()

    video_exists = []
    for vid_key in ds_meta.video_keys:
        video_path = str(video_root / ds_meta.get_video_file_path(ep_idx, vid_key))
        video_exists.append(os.path.exists(video_path))
    
    if not all(video_exists):
        print("[INFO] video file not found for episode: {}".format(ep_idx))
        continue

    save_path = str(h5_droid_root / parquet_file).replace(".parquet", ".h5")
    if opt.skip_saved and os.path.exists(save_path):
        pass

    timestamp = torch.stack(hf_dataset["timestamp"]).cpu().numpy()  # (T,)
    wrist_left_cam_poses = torch.stack(hf_dataset["camera_extrinsics.wrist_left"]).cpu().numpy()  # (T, 6)
    wrist_left_cam_poses = xyzrpy2mat(wrist_left_cam_poses)  # (T, 4, 4)
    
    exterior_1_left_cam_poses = torch.stack(hf_dataset["camera_extrinsics.exterior_1_left"]).cpu().numpy()  # (T, 6)
    exterior_1_left_cam_poses = xyzrpy2mat(exterior_1_left_cam_poses)  # (T, 4, 4)
    
    exterior_2_left_cam_poses = torch.stack(hf_dataset["camera_extrinsics.exterior_2_left"]).cpu().numpy()  # (T, 6)
    exterior_2_left_cam_poses = xyzrpy2mat(exterior_2_left_cam_poses)  # (T, 4, 4)
    
    ee_poses = torch.stack(hf_dataset["observation.state.cartesian_position"]).cpu().numpy()
    ee_poses = xyzrpy2mat(ee_poses)
    
    # modify wrist cam pose, which keeps the same relative pose with ee as the first frame
    ecT = np.linalg.inv(ee_poses[0]) @ wrist_left_cam_poses[0]
    wrist_left_cam_poses: np.ndarray = ee_poses @ ecT
    ########################
    
    # lerobot use 0 for open, 1 for close. We use 0 for close, 1 for open.
    grippers = 1 - torch.stack(hf_dataset["observation.state.gripper_position"]).cpu().numpy()
    grippers_desired = 1 - torch.stack(hf_dataset["action.gripper_position"]).cpu().numpy()
    
    # # filter out noops
    # actions = torch.stack(hf_dataset["action"]).cpu().numpy()
    # delta_actions = actions[1:] - actions[:-1]
    # delta_actions_norm = np.linalg.norm(delta_actions, axis=-1)
    # hasop_mask = delta_actions_norm > 1e-3
    # hasop_mask = np.concatenate([[True], hasop_mask])
    
    lang1 = hf_dataset["language_instruction"][0]
    lang2 = hf_dataset["language_instruction_2"][0]
    lang3 = hf_dataset["language_instruction_3"][0]
    is_terminal = torch.stack(hf_dataset["is_terminal"]).cpu().numpy()

    data_dict = {
        "timestamp": timestamp.astype(np.float32), 
        "ee_pose": ee_poses.astype(np.float32),
        "gripper": grippers.astype(np.float32),
        "gripper_desired": grippers_desired.astype(np.float32),

        "wrist_left/K": default_K.astype(np.float32),
        "wrist_left/rgb": [str(ds_meta.get_video_file_path(ep_idx, "observation.images.wrist_left"))],
        "wrist_left/pose": wrist_left_cam_poses.astype(np.float32),

        "exterior_1_left/K": default_K.astype(np.float32),
        "exterior_1_left/rgb": [str(ds_meta.get_video_file_path(ep_idx, "observation.images.exterior_1_left"))],
        "exterior_1_left/pose": exterior_1_left_cam_poses.astype(np.float32),

        "exterior_2_left/K": default_K.astype(np.float32),
        "exterior_2_left/rgb": [str(ds_meta.get_video_file_path(ep_idx, "observation.images.exterior_2_left"))],
        "exterior_2_left/pose": exterior_2_left_cam_poses.astype(np.float32),
    }

    data_dict = {k:np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v 
                 for k, v in data_dict.items()}

    attr_dict = {
        "compress": "mp4",
        "episode_index": ep_idx,
        "prompt_text": lang1,
        "prompt_text2": lang2,
        "prompt_text3": lang3,
    }

    
    save_to_h5(
        save_path,
        data_dict=data_dict,
        attr_dict=attr_dict
    )
    print("[INFO] file saved to {}".format(save_path))
    print("[INFO] episode: {}/{}".format(ep_idx, ds_meta.total_episodes))
    # print("[INFO] success: {}".format(success))
    print("[INFO] lang1: {}".format(lang1))
    print("[INFO] lang2: {}".format(lang2))
    print("[INFO] lang3: {}".format(lang3))
    # print("[INFO] terminal: {}".format(is_terminal.astype(np.uint8)))
    # print("[INFO] skip {}/{} frames".format(len(hasop_mask) - hasop_mask.sum(), len(hasop_mask)))
    
    t1 = time.time()
    print("[INFO] iter time: {}".format(t1 - t0))

