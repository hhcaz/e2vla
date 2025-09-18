import os
import cv2
import h5py
import json
import tqdm
import argparse
import traceback
import numpy as np
from einops import rearrange
from scipy.spatial.transform import Rotation

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2mat, quat2axisangle
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix


IMAGE_RESOLUTION = 256


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    # env = OffScreenRenderEnv(**env_args, camera_depths=True)
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def proj(K: np.ndarray, cwT: np.ndarray, pos: np.ndarray):
    pos_in_cam = pos @ cwT[:3, :3].T + cwT[:3, 3]
    xy = pos_in_cam[:2] / pos_in_cam[-1:]

    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    fxy = np.array([fx, fy]); cxy = np.array([cx, cy])
    uv = xy * fxy + cxy
    return uv


def draw_ee_axis(bgr: np.ndarray, K: np.ndarray, cwT: np.ndarray, pose: np.ndarray):
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


def vis_regenerate_obs(
    obs: dict, 
    eef_pos: np.ndarray, 
    eef_rotmat: np.ndarray, 
    sim, 
    action=None, 
    env=None
):
    cam_K_e2h = get_camera_intrinsic_matrix(sim, "agentview", IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    cam_K_eih = get_camera_intrinsic_matrix(sim, "robot0_eye_in_hand", IMAGE_RESOLUTION, IMAGE_RESOLUTION)

    cam_wcT_e2h = get_camera_extrinsic_matrix(sim, "agentview")
    cam_wcT_eih = get_camera_extrinsic_matrix(sim, "robot0_eye_in_hand")

    act_img_e2h = proj(cam_K_e2h, np.linalg.inv(cam_wcT_e2h), eef_pos)
    act_img_eih = proj(cam_K_eih, np.linalg.inv(cam_wcT_eih), eef_pos)

    rgb_e2h = obs["agentview_image"]
    rgb_eih = obs["robot0_eye_in_hand_image"]

    ### Why upside down????????
    bgr_e2h = np.ascontiguousarray(rgb_e2h[::-1, :, [2, 1, 0]])
    bgr_eih = np.ascontiguousarray(rgb_eih[::-1, :, [2, 1, 0]])

    cv2.circle(bgr_e2h, tuple(act_img_e2h.astype(int).tolist()), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(bgr_eih, tuple(act_img_eih.astype(int).tolist()), radius=2, color=(0, 0, 255), thickness=-1)

    eef_pose = np.eye(4).astype(eef_pos.dtype); eef_pose[:3, :3] = eef_rotmat; eef_pose[:3, 3] = eef_pos
    draw_ee_axis(bgr_e2h, cam_K_e2h, np.linalg.inv(cam_wcT_e2h), eef_pose)
    draw_ee_axis(bgr_eih, cam_K_eih, np.linalg.inv(cam_wcT_eih), eef_pose)

    if action is not None:
        future_eef_pose = delta_action_to_future_pose(eef_pos, eef_rotmat, action, env.env.robots[0].controller.action_scale)
        draw_ee_axis(bgr_e2h, cam_K_e2h, np.linalg.inv(cam_wcT_e2h), future_eef_pose)
        draw_ee_axis(bgr_eih, cam_K_eih, np.linalg.inv(cam_wcT_eih), future_eef_pose)

    bgr = np.concatenate([bgr_e2h, bgr_eih], axis=1)
    cv2.imshow("regenerate", bgr)
    key = cv2.waitKey(1)
    return key


def vis_original_obs(agent_view_rgb: np.ndarray, eye_in_hand_rgb: np.ndarray):
    ### Why upside down????????
    agent_view_bgr = np.ascontiguousarray(agent_view_rgb[::-1, :, [2, 1, 0]])
    eye_in_hand_bgr = np.ascontiguousarray(eye_in_hand_rgb[::-1, :, [2, 1, 0]])
    bgr = np.concatenate([agent_view_bgr, eye_in_hand_bgr], axis=1)
    cv2.imshow("original", bgr)
    key = cv2.waitKey(1)
    return key


def jpeg_encode(image: np.ndarray):
    return np.frombuffer(cv2.imencode(".jpg", image)[1].data, dtype=np.uint8)


def jpeg_decode(array: np.ndarray):
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def transform_image(rgbs: np.ndarray, compress: bool):
    rgbs = np.ascontiguousarray(rgbs[:, ::-1, :, :])  # flip the height dimension
    if not compress:
        return rearrange(rgbs, "n h w c -> n c h w")
    else:
        return [jpeg_encode(rgb) for rgb in rgbs]


def libero2ours(
    actions: np.ndarray, 
    gripper_states: np.ndarray,
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,

    e2h_rgb: np.ndarray,
    e2h_pose: np.ndarray,
    e2h_K: np.ndarray,

    eih_rgb: np.ndarray,
    eih_pose: np.ndarray,
    eih_K: np.ndarray,
    
    compress: bool,

    robosuite_action_scale: np.ndarray
):
    """
    gripper_states: (Ta, 2)
    ee_pos: (Ta, 3)
    ee_ori: (Ta, 3)

    e2h_rgb: (Ta, H, W, 3)
    e2h_pose: (Ta, 4, 4)
    e2h_K: (3, 3)

    eih_rgb: (Ta, H, W, 3)
    eih_pose: (Ta, 4, 4)
    eih_K: (3, 3)

    compress: bool
    """
    
    open_gripper_qpos = 0.04
    close_gripper_qpos = 0.0

    Ta = gripper_states.shape[0]
    gripper = (gripper_states[:, 0] - close_gripper_qpos) / (open_gripper_qpos - close_gripper_qpos)
    gripper_desired = (-actions[:, -1] + 1) / 2.0

    ee_pose = np.zeros((Ta, 4, 4), dtype=ee_pos.dtype)
    ee_pose[:, :3, :3] = Rotation.from_rotvec(ee_ori).as_matrix()
    ee_pose[:, :3, 3] = ee_pos
    ee_pose[:, 3, 3] = 1

    ee_pose_desired = delta_action_to_future_pose(
        current_eef_pos=ee_pose[:, :3, 3],
        current_eef_rotmat=ee_pose[:, :3, :3],
        delta_eef_action=actions,
        robosuite_action_scale=robosuite_action_scale
    )

    data_dict = {
        "ee_pose": ee_pose.astype(np.float32),
        "gripper": gripper.astype(np.float32),
        "ee_pose_desired": ee_pose_desired.astype(np.float32), 
        "gripper_desired": gripper_desired.astype(np.float32)
    }

    data_dict["agentview/rgb"] = transform_image(e2h_rgb, compress)
    data_dict["agentview/pose"] = e2h_pose.astype(np.float32)
    data_dict["agentview/K"] = e2h_K.astype(np.float32)  # (3, 3)

    data_dict["eye_in_hand/rgb"] = transform_image(eih_rgb, compress)
    data_dict["eye_in_hand/pose"] = eih_pose.astype(np.float32)
    data_dict["eye_in_hand/K"] = eih_K.astype(np.float32)  # (3, 3)

    data_dict["timestamp"] = np.arange(Ta).astype(np.float32)

    return data_dict


def delta_action_to_future_pose(
    current_eef_pos: np.ndarray,        # (..., 3)
    current_eef_rotmat: np.ndarray,     # (..., 3, 3)
    delta_eef_action: np.ndarray,       # (..., 7)
    robosuite_action_scale: np.ndarray
):
    # https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1_libero/robosuite/utils/control_utils.py#L150
    # goal = delta_rot @ current  # Why left multiply??????????

    # actions are scaled: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1_libero/robosuite/controllers/osc.py#L237

    delta_pos = delta_eef_action[..., 0:3] * robosuite_action_scale[0:3]
    future_eef_pos = current_eef_pos + delta_pos

    delta_rotmat = Rotation.from_rotvec(delta_eef_action[..., 3:6] * robosuite_action_scale[3:6]).as_matrix()
    future_eef_rotmat = delta_rotmat @ current_eef_rotmat

    future_ee_pose = np.eye(4)
    if current_eef_pos.ndim > 1:
        future_ee_pose = future_ee_pose[None].repeat(len(current_eef_pos), axis=0)

    future_ee_pose[..., :3, :3] = future_eef_rotmat
    future_ee_pose[..., :3, 3] = future_eef_pos
    return future_ee_pose



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


def check_data_integrity(path: str):
    if not os.path.exists(path):
        return False
    
    integrity = True
    try:
        with h5py.File(path, mode="r") as h5:
            for k in ["ee_pose", "gripper", "ee_pose_desired", "gripper_desired", "timestamp"]:
                data = h5[k][:]
            for k1 in ["agentview", "eye_in_hand"]:
                for k2 in ["rgb", "pose", "K"]:
                    data = h5[k1][k2][0]
    except Exception as e:
        traceback.print_exc()
        integrity = False
    return integrity



def main(args):
    ENABLE_VISUALIZE = args.visualize and ("DISPLAY" in os.environ)
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # # Create target directory
    # if os.path.isdir(args.libero_target_dir):
    #     user_input = input(f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
    #     if user_input != 'y':
    #         exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"{args.libero_target_dir}/{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        new_data_dir = os.path.join(args.libero_target_dir, task.name)

        for i in range(len(orig_data.keys())):
            print(f"[INFO] task_id = {task_id}")
            print(f"[INFO] demo_id = {i}")
            print(f"[INFO] task_description = {task_description}")
            new_data_path = os.path.join(new_data_dir, "demo_{:0>4d}.h5".format(i))

            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            orig_obs = demo_data["obs"]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action())
                if ENABLE_VISUALIZE:
                    vis_regenerate_obs(obs, obs["robot0_eef_pos"], quat2mat(obs["robot0_eef_quat"]), env.sim)

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            agentview_poses = []
            eye_in_hand_poses = []
            agentview_K = get_camera_intrinsic_matrix(env.sim, "agentview", IMAGE_RESOLUTION, IMAGE_RESOLUTION)
            eye_in_hand_K = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", IMAGE_RESOLUTION, IMAGE_RESOLUTION)

            # Replay original demo actions in environment and record observations
            for a_idx, action in enumerate(orig_actions):

                # action: https://github.com/ARISE-Initiative/robosuite/issues/139
                # Behavior of osc_pose controller with control_delta = False #139
                # https://robosuite.ai/docs/modules/controllers.html
                if ENABLE_VISUALIZE:
                    vis_original_obs(orig_obs["agentview_rgb"][a_idx], orig_obs["eye_in_hand_rgb"][a_idx])

                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                agentview_poses.append(get_camera_extrinsic_matrix(env.sim, "agentview"))
                eye_in_hand_poses.append(get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand"))

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())
                if ENABLE_VISUALIZE:
                    vis_regenerate_obs(
                        obs, obs["robot0_eef_pos"], quat2mat(obs["robot0_eef_quat"]), env.sim,
                        action=action, env=env)

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                data_dict = libero2ours(
                    actions=np.stack(actions, axis=0),
                    gripper_states=np.stack(gripper_states, axis=0).astype(np.float32),
                    ee_pos=np.stack(ee_states, axis=0)[:, :3].astype(np.float32),
                    ee_ori=np.stack(ee_states, axis=0)[:, 3:].astype(np.float32),

                    e2h_rgb=np.stack(agentview_images, axis=0),
                    e2h_pose=np.stack(agentview_poses, axis=0).astype(np.float32),
                    e2h_K=agentview_K.astype(np.float32),

                    eih_rgb=np.stack(eye_in_hand_images, axis=0),
                    eih_pose=np.stack(eye_in_hand_poses, axis=0).astype(np.float32),
                    eih_K=eye_in_hand_K.astype(np.float32),

                    compress=True,
                    robosuite_action_scale=env.env.robots[0].controller.action_scale
                )

                write_to_h5(
                    data_dict=data_dict,
                    attr_dict={
                        # "prompt_text": task.name.replace("_", " "),
                        "prompt_text": task_description, 
                        "compress": True,
                    },
                    h5_path=new_data_path
                )
                print("[INFO] data write to: {}".format(new_data_path))
                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        # new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets",
                        default="./data_raw/libero")
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets",
                        default="./data_converted/libero")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--skip_saved", action="store_true", default=False)
    args = parser.parse_args()

    args.libero_raw_data_dir = os.path.join(args.libero_raw_data_dir, args.libero_task_suite)
    args.libero_target_dir = os.path.join(args.libero_target_dir, args.libero_task_suite + "_no_noops")

    # Start data regeneration
    main(args)
