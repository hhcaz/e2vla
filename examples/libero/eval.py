import os
import cv2
import tqdm
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path
from robosuite.utils.transform_utils import quat2mat
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

from data_utils import align
from .video_writer import VideoWriter
from shm_transport import get_shm_proxy, setup_log_level


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


def proj(K: np.ndarray, cwT: np.ndarray, pos: np.ndarray):
    pos_in_cam = pos @ cwT[:3, :3].T + cwT[:3, 3]
    xy = pos_in_cam[..., :2] / pos_in_cam[..., -1:]

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


def vis_regen_obs(obs: dict, eef_pos: np.ndarray, eef_rotmat: np.ndarray, sim, pred_poses = None):
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
    draw_ee_proj(bgr_e2h, cam_K_e2h, np.linalg.inv(cam_wcT_e2h), eef_pose)
    draw_ee_proj(bgr_eih, cam_K_eih, np.linalg.inv(cam_wcT_eih), eef_pose)
    
    if pred_poses is not None:
        future_act_e2h = proj(cam_K_e2h, np.linalg.inv(cam_wcT_e2h), pred_poses[..., :3, 3])
        for pts in future_act_e2h:
            cv2.circle(bgr_e2h, tuple(pts.astype(int).tolist()), radius=2, color=(0, 0, 255), thickness=-1)

    bgr = np.concatenate([bgr_e2h, bgr_eih], axis=1)
    cv2.imshow("eval_pred", bgr)
    key = cv2.waitKey(1)
    return bgr, key


def action_ours2libero(
    future_ee_poses: np.ndarray,  # (T, 4, 4)
    future_grippers: np.ndarray,  # (T,)
    current_eef_pos: np.ndarray,  # (3,)
    current_eef_rotmat: np.ndarray,  # (3, 3)
    robosuite_action_scale: np.ndarray
):
    delta_pos = future_ee_poses[..., :3, 3] - current_eef_pos

    # https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1_libero/robosuite/utils/control_utils.py#L150
    # goal = delta_rot @ current  # Why left multiply??????????
    delta_rotmat = future_ee_poses[..., :3, :3] @ np.linalg.inv(current_eef_rotmat)
    delta_ori = Rotation.from_matrix(delta_rotmat).as_rotvec()

    # https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1_libero/robosuite/controllers/osc.py#L237
    delta_pose = np.concatenate([delta_pos, delta_ori], axis=-1)  # (Ta, 6)
    delta_pose_scaled = delta_pose / robosuite_action_scale  # action is scaled
    
    binary_future_grippers = (future_grippers > 0.5).astype(np.float32)  # 0 for close, 1 for open
    delta_gripper = 1 - 2 * binary_future_grippers
    delta_action = np.concatenate([delta_pose_scaled, delta_gripper[..., None]], axis=-1)  # (Ta, 7)
    return delta_action


def obs_libero2ours(obs: dict, time, env):

    ee_pose = np.eye(4)
    ee_pose[:3, 3] = obs["robot0_eef_pos"]
    ee_pose[:3, :3] = quat2mat(obs["robot0_eef_quat"])

    gripper_state = obs["robot0_gripper_qpos"]  # (2,)
    open_gripper_qpos = 0.04
    close_gripper_qpos = 0.0
    gripper = (gripper_state[0] - close_gripper_qpos) / (open_gripper_qpos - close_gripper_qpos)

    e2h_rgb = np.ascontiguousarray(obs["agentview_image"][::-1])  # flip H
    e2h_pose = get_camera_extrinsic_matrix(env.sim, "agentview")
    e2h_K = get_camera_intrinsic_matrix(env.sim, "agentview", IMAGE_RESOLUTION, IMAGE_RESOLUTION)

    eih_rgb = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    eih_pose = get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")
    eih_K = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", IMAGE_RESOLUTION, IMAGE_RESOLUTION)

    frame_dict = {
        "ee_pose": ee_pose,
        "gripper": gripper,
        "timestamp": time,
        "agentview": {
            "model": "pinhole",
            "camera": {"height": IMAGE_RESOLUTION, "width": IMAGE_RESOLUTION, "K": e2h_K},
            "data": {"color": e2h_rgb, "wcT": e2h_pose, "seg": None, "depth": None}
        },
        "eye_in_hand": {
            "model": "pinhole",
            "camera": {"height": IMAGE_RESOLUTION, "width": IMAGE_RESOLUTION, "K": eih_K},
            "data": {"color": eih_rgb, "wcT": eih_pose, "seg": None, "depth": None}
        }
    }
    return frame_dict


def modify_prompt(lang: str):
    # remove something like "LIVING ROOM SCENE6" in libero10 and libero90
    index = lang.find("SCENE")
    if index >= 0:
        lang = lang[index:]
        lang = " ".join(lang.split(" ")[1:])
    return lang


def main(args):
    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    max_eval_ep_len = 500

    controller = get_shm_proxy(
        args.uri, 
        ns_host=args.ns_host,
        ns_port=args.ns_port,
    )
    setup_log_level("WARNING")
    
    runtime_config = controller.get_config()
    # runtime_config["camera_names"] = runtime_config["camera_names"][::-1]
    # controller.set_config(runtime_config)
    # print("[INFO] set config to: {}".format(runtime_config))

    print("!"*101)
    print("[INFO] task_suite = {}, uri = {}"
          .format(args.task_suite, args.uri))
    input("[INFO] Press Enter to continue: ")

    save_path = f"./eval_results/{args.task_suite}/{args.uri}.csv"
    video_root = f"./eval_videos/{args.task_suite}/{args.uri}"
    
    if args.save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fp = open(save_path, "w")
        fp.write("task_id, episode_id, success, prompt\n")
        fp.flush()
    
    if args.video:
        os.makedirs(video_root, exist_ok=True)
    
    success_records = []

    # prev_task_name = None
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = get_libero_env(task, resolution=IMAGE_RESOLUTION)

        print("[INFO] task_description = {}".format(task_description))

        for i in range(args.rounds):
            save_video_path = os.path.join(video_root, "task{:0>3d}".format(task_id), "ep{:0>3d}.mp4".format(i))
            if args.video:
                os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
                vid_writer = VideoWriter(save_video_path, 30)

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(initial_states[i])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action())
                vis_regen_obs(obs, obs["robot0_eef_pos"], quat2mat(obs["robot0_eef_quat"]), env.sim)

            prompt_text = task.name.replace("_", " ")
            prompt_text = modify_prompt(prompt_text)
            print("prompt_text = {}".format(prompt_text))

            controller.reset()
            sample_state_gaps = runtime_config["sample_state_gaps"]

            a_idx = 0
            controller.set_ensemble_nums(4)
            # controller.set_ensemble_nums(0)  # disables ensemble
            controller.set_prompt(prompt_text)
            controller.add_obs_frame(obs_libero2ours(obs, time=a_idx, env=env))
            print("[INFO] prompt_text = {}".format(prompt_text))

            while True:
                future_ee_poses, future_grippers, future_time, _ = controller.get_action()
                # future_ee_poses: (Ta, Nee, 4, 4), future_grippers: (Ta, Nee)
                # valid_ee_indices = (0,), therefore we fetch the first 
                future_ee_poses = future_ee_poses[:, 0]
                future_grippers = future_grippers[:, 0]
                
                # exec action ensemble
                future_ee_poses, future_grippers = controller.ensemble_traj(
                    future_ee_poses, future_grippers, future_time
                )

                if sample_state_gaps != 1:
                    query_time = a_idx + np.arange(len(future_ee_poses) * sample_state_gaps)
                    train_data = {"ee_pose": future_ee_poses, "gripper": future_grippers}
                    interp_funcs = {"ee_pose": align.interp_SE3_sep, "gripper": align.interp_linear}
                    query_data = align.align_data(query_time, future_time, train_data, interp_funcs)
                else:
                    query_data = {"ee_pose": future_ee_poses, "gripper": future_grippers}

                for s_id in range(3):
                    action = action_ours2libero(
                        future_ee_poses=query_data["ee_pose"][s_id],
                        future_grippers=query_data["gripper"][s_id],
                        current_eef_pos=obs["robot0_eef_pos"],
                        current_eef_rotmat=quat2mat(obs["robot0_eef_quat"]),
                        robosuite_action_scale=env.env.robots[0].controller.action_scale
                    )

                    # action: https://github.com/ARISE-Initiative/robosuite/issues/139
                    # Behavior of osc_pose controller with control_delta = False #139
                    # https://robosuite.ai/docs/modules/controllers.html

                    # Execute demo action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    a_idx += 1

                    debug_bgr, key = vis_regen_obs(
                        obs, obs["robot0_eef_pos"], quat2mat(obs["robot0_eef_quat"]), env.sim,
                        pred_poses=query_data["ee_pose"])
                    controller.add_obs_frame(obs_libero2ours(obs, time=a_idx, env=env))
                    if args.video:
                        vid_writer.write(debug_bgr)

                    if done:
                        break
                    if a_idx > max_eval_ep_len:
                        break
                    if key == ord('n'):
                        break
                
                if done:
                    break
                if a_idx > max_eval_ep_len:
                    break
                if key == ord('n'):
                    break
            
            if args.video:
                vid_writer.finalize()

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                num_success += 1

            num_replays += 1

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")
            # break

            if args.save:
                fp.write(f"{task_id}, {i}, {int(done)}, {prompt_text}\n")
                fp.flush()
            success_records.append(int(done))

            # if not args.all:
            #     break  # only use first existing demo in each unique demonstration

    if args.save:
        fp.write(" , , {:.2f}%, \n".format(np.mean(success_records) * 100))
        fp.close()



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--rounds", type=int, default=50)
    # find the hosting obect
    parser.add_argument("--uri", type=str, default="e2vla")
    parser.add_argument("--ns_host", type=str, default="localhost")
    parser.add_argument("--ns_port", type=int, default=9090)
    #####
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
