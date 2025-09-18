import os
import cv2
import h5py
import argparse
import numpy as np
from einops import rearrange
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation


def jpeg_encode(image: np.ndarray):
    return np.frombuffer(cv2.imencode(".jpg", image)[1].data, dtype=np.uint8)


def jpeg_decode(array: np.ndarray):
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def proj(K: np.ndarray, wcT: np.ndarray, pos: np.ndarray):
    cwT = np.linalg.inv(wcT)
    pos_in_cam = pos @ cwT[:3, :3].T + cwT[:3, 3]
    xy = pos_in_cam[:2] / pos_in_cam[-1:]

    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    fxy = np.array([fx, fy]); cxy = np.array([cx, cy])
    uv = xy * fxy + cxy
    return uv


def draw_ee_axis(rgb: np.ndarray, K: np.ndarray, cwT: np.ndarray, pose: np.ndarray):
    wcT = np.linalg.inv(cwT)
    
    alen = 0.05  # 5cm
    pos = pose[:3, 3]
    x_end = tuple(proj(K, wcT, pos + pose[:3, 0] * alen).astype(int).tolist())
    y_end = tuple(proj(K, wcT, pos + pose[:3, 1] * alen).astype(int).tolist())
    z_end = tuple(proj(K, wcT, pos + pose[:3, 2] * alen).astype(int).tolist())
    origin = tuple(proj(K, wcT, pos).astype(int).tolist())
    
    bgr = np.ascontiguousarray(rgb[:, :, [2, 1, 0]])
    cv2.line(bgr, origin, x_end, (0, 0, 255), thickness=2)
    cv2.line(bgr, origin, y_end, (0, 255, 0), thickness=2)
    cv2.line(bgr, origin, z_end, (255, 0, 0), thickness=2)
    
    return bgr


def pq2mat(pq: np.ndarray):
    pos = pq[:3]
    quat = pq[3:]  # (qw, qx, qy, qz)
    quat = np.concatenate([quat[1:], quat[0:1]], axis=-1)  # (qx, qy, qz, qw)
    
    T = np.eye(4, dtype=pq.dtype)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def norm_gripper_qpos(q):
    qmax = 0.04
    qmin = 0.0
    qnorm = (q - qmin) / (qmax - qmin)
    return qnorm


def norm_gripper_qpos_d(q_d):
    return (q_d + 1) / 2.0  # -1,1 -> 0,1: 0 fully close 1 fully open


def write_to_h5(
    data_dict: dict,
    attr_dict: dict, 
    h5_path: str
):
    output_dir = os.path.dirname(h5_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, "w") as h5:
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                h5.create_dataset(k, data=v)
            else:
                dtype = h5py.vlen_dtype(v[0].dtype)
                dset = h5.create_dataset(k, shape=(len(v),), dtype=dtype)
                for i, vi in enumerate(v):
                    dset[i] = vi

        for k, v in attr_dict.items():
            h5.attrs[k] = v




parser = argparse.ArgumentParser()
parser.add_argument("--input_root", type=str, default="./data_raw/maniskill/0.1.0")
parser.add_argument("--output_root", type=str, default="./data_converted/maniskill/0.1.0")
parser.add_argument("--visualize", action="store_true", default=False)
opt = parser.parse_args()


b = tfds.builder_from_directory(builder_dir=opt.input_root)
ds = b.as_dataset(split='train')
save_dir = opt.output_root

visualize = opt.visualize and ("DISPLAY" in os.environ)


if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)


num_ep = 0
for ep_id, episode in enumerate(ds):
    num_ep += 1  # total 30213 episodes
    
    print("-"*61)
    print("[INFO] ep = {}".format(ep_id))

    data_dict = {
        "ee_pose": [],
        "gripper": [],
        "gripper_desired": [],
        "timestamp": [],

        "main_camera/rgb": [],
        "main_camera/pose": [],
        "main_camera/K": None,

        "wrist_camera/rgb": [],
        "wrist_camera/pose": [],
        "wrist_camera/K": None,
    }
    attr_dict = {}
    compress = True
    
    for i, step in enumerate(episode["steps"]):
        if i == 0:
            print("[INFO] instruction = {}".format(step["language_instruction"].numpy().decode("utf-8")))
            attr_dict["compress"] = compress
            attr_dict["prompt_text"] = step["language_instruction"].numpy().decode("utf-8")

            main_camera_K = step["observation"]["main_camera_intrinsic_cv"].numpy()
            wrist_camera_K = step["observation"]["wrist_camera_intrinsic_cv"].numpy()

            data_dict["main_camera/K"] = main_camera_K.astype(np.float32)
            data_dict["wrist_camera/K"] = wrist_camera_K.astype(np.float32)
        
        main_camera_rgb = step["observation"]["image"].numpy()
        main_camera_cwT = step["observation"]["main_camera_extrinsic_cv"].numpy()
        data_dict["main_camera/rgb"].append(jpeg_encode(main_camera_rgb) if compress else rearrange(main_camera_rgb, "h w c -> c h w"))
        data_dict["main_camera/pose"].append(np.linalg.inv(main_camera_cwT).astype(np.float32))
        
        wrist_camera_rgb = step["observation"]["wrist_image"].numpy()
        wrist_camera_cwT = step["observation"]["wrist_camera_extrinsic_cv"].numpy()
        data_dict["wrist_camera/rgb"].append(jpeg_encode(wrist_camera_rgb) if compress else rearrange(wrist_camera_rgb, "h w c -> c h w"))
        data_dict["wrist_camera/pose"].append(np.linalg.inv(wrist_camera_cwT).astype(np.float32))
        
        ee_pose = pq2mat(step["observation"]["tcp_pose"].numpy())
        gripper_qpos = step["observation"]["state"].numpy()[7:9]  # (2,)
        gripper_qpos_d = step["action"].numpy()[-1]

        data_dict["ee_pose"].append(ee_pose)
        data_dict["gripper"].append(norm_gripper_qpos(gripper_qpos[0]))
        data_dict["gripper_desired"].append(norm_gripper_qpos_d(gripper_qpos_d))
        data_dict["timestamp"].append(i)

        if visualize:
            debug_bgr = np.concatenate([
                draw_ee_axis(main_camera_rgb, main_camera_K, main_camera_cwT, ee_pose),
                draw_ee_axis(wrist_camera_rgb, wrist_camera_K, wrist_camera_cwT, ee_pose),
            ], axis=1)
            
            cv2.imshow("main | wrist", debug_bgr)
            key = cv2.waitKey(1)
            if key == ord('q'):
                quit()
    
    for k, v in data_dict.items():
        if isinstance(v, list):
            if ("rgb" in k) and compress:
                continue
            else:
                data_dict[k] = np.stack(v, axis=0)
    
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            print("- k = {}, shape = {}".format(k, v.shape))
        else:
            print("- k = {}, len = {}".format(k, len(v)))

    save_path = os.path.join(save_dir, "{:0>5d}.h5".format(ep_id))
    write_to_h5(
        data_dict=data_dict,
        attr_dict=attr_dict,
        h5_path=save_path
    )
    print("[INFO] data saved to {}".format(save_path))

