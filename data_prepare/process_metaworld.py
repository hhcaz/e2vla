import os
os.environ["MUJOCO_GL"] = "egl"

import cv2
import h5py
import mujoco
import argparse
import metaworld
import gymnasium
import numpy as np
import metaworld.policies
from tqdm import tqdm
from einops import rearrange
from metaworld.sawyer_xyz_env import SawyerXYZEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


def put_text(image: np.ndarray, text=[], small_text=[]):
    H = image.shape[0]
    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(image, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(image, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(image, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(image, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
    return image


def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera intrinsic matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    # cam_id = sim.model.camera_name2id(camera_name)
    cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(sim, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        T (np.array): 4x4 camera extrinsic matrix
    """
    # cam_id = sim.model.camera_name2id(camera_name)
    cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)

    T = np.eye(4)
    T[:3, :3] = camera_rot
    T[:3, 3] = camera_pos

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    T = T @ camera_axis_correction
    return T


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


def get_eef_rot(env):
    return env.data.body("hand").xmat.reshape(3, 3)


def get_eef_pose(env) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = get_eef_rot(env)
    T[:3, 3] = env.tcp_center
    return T


class MyEnvWrapper(gymnasium.Wrapper):
    DEFAULT_SIZE = 256
    DEFAULT_CAMERA_CONFIG = None
    CAMERA_NAMES = ['corner', 'corner2', 'corner3', 'topview', 'behindGripper', 'gripperPOV']

    def __init__(
        self, 
        env: SawyerXYZEnv, 
        # seed: int = None,
        max_geom: int = 1000,
    ):
        super().__init__(env)
        self.env: SawyerXYZEnv

        self.unwrapped.model.vis.global_.offwidth = self.DEFAULT_SIZE
        self.unwrapped.model.vis.global_.offheight = self.DEFAULT_SIZE

        self.unwrapped.cam_corner = MujocoRenderer(
            self.unwrapped.model, self.unwrapped.data, self.DEFAULT_CAMERA_CONFIG, self.DEFAULT_SIZE, self.DEFAULT_SIZE, max_geom,
            camera_id=None, camera_name="corner")
        
        self.unwrapped.cam_corner2 = MujocoRenderer(
            self.unwrapped.model, self.unwrapped.data, self.DEFAULT_CAMERA_CONFIG, self.DEFAULT_SIZE, self.DEFAULT_SIZE, max_geom,
            camera_id=None, camera_name="corner2")
        
        self.unwrapped.cam_corner3 = MujocoRenderer(
            self.unwrapped.model, self.unwrapped.data, self.DEFAULT_CAMERA_CONFIG, self.DEFAULT_SIZE, self.DEFAULT_SIZE, max_geom,
            camera_id=None, camera_name="corner3")

        self.unwrapped.cam_topview = MujocoRenderer(
            self.unwrapped.model, self.unwrapped.data, self.DEFAULT_CAMERA_CONFIG, self.DEFAULT_SIZE, self.DEFAULT_SIZE, max_geom,
            camera_id=None, camera_name="topview")
        
        self.unwrapped.cam_behindGripper = MujocoRenderer(
            self.unwrapped.model, self.unwrapped.data, self.DEFAULT_CAMERA_CONFIG, self.DEFAULT_SIZE, self.DEFAULT_SIZE, max_geom,
            camera_id=None, camera_name="behindGripper")
        
        self.unwrapped.cam_gripperPOV = MujocoRenderer(
            self.unwrapped.model, self.unwrapped.data, self.DEFAULT_CAMERA_CONFIG, self.DEFAULT_SIZE, self.DEFAULT_SIZE, max_geom,
            camera_id=None, camera_name="gripperPOV")

        # Hack: enable random reset
        self.unwrapped._freeze_rand_vec = False
        # if seed is not None:
        #     self.unwrapped.seed(seed)x
    
    def render(self):
        results = {}

        for name in self.CAMERA_NAMES:
            renderer: MujocoRenderer = getattr(self.env, "cam_" + name)
            rgb = renderer.render("rgb_array")

            results[name] = {
                "rgb": rgb,
                "K": get_camera_intrinsic_matrix(self.env, name, self.DEFAULT_SIZE, self.DEFAULT_SIZE),
                "wcT": get_camera_extrinsic_matrix(self.env, name)
            }

        return results
    
    def close(self):
        for name in self.CAMERA_NAMES:
            renderer: MujocoRenderer = getattr(self.env, "cam_" + name)
            renderer.close()
        
        return super().close()

    @property
    def eef_pose(self):
        return get_eef_pose(self.env)


def rotate180(frame: dict):
    frame["rgb"] = np.ascontiguousarray(frame["rgb"][::-1, ::-1])
    frame["wcT"][:3, :3] = frame["wcT"][:3, :3] @ np.array([[-1., 0., 0.],
                                                            [0., -1, 0.],
                                                            [0., 0., 1.]])
    return frame


def jpeg_encode(image: np.ndarray):
    return np.frombuffer(cv2.imencode(".jpg", image)[1].data, dtype=np.uint8)


def jpeg_decode(array: np.ndarray):
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def stack_or_compress(rgbs: np.ndarray, compress: bool):
    if not compress:
        return rearrange(rgbs, "n h w c -> n c h w")
    else:
        return [jpeg_encode(rgb) for rgb in rgbs]


def make_frame_dict(
    cam_frames: dict, 
    eef_pose: np.ndarray, 
    gripper: float, 
    timestamp: float,
    gripper_action: float = None,
):
    frame_dict = {
        "ee_pose": eef_pose,
        "gripper": gripper,
        "timestamp": timestamp,
    }
    
    if gripper_action is not None:
        frame_dict["gripper_desired"] = -gripper_action/2.0 + 0.5
    
    for cam_name, cam_frame in cam_frames.items():
        frame_dict[cam_name] = {
            "model": "pinhole",
            "camera": {
                "height": cam_frame["rgb"].shape[0], 
                "width": cam_frame["rgb"].shape[1],
                "K": cam_frame["K"]
            },
            "data": {
                "color": cam_frame["rgb"],
                "wcT": cam_frame["wcT"],
                "depth": None,
                "seg": None,
            }
        }
    
    return frame_dict


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


def run_env(env: MyEnvWrapper, task_name: str, max_ep_len: int = 500):
    obs, info = env.reset()
    policy = metaworld.policies.ENV_POLICY_MAP[task_name]()

    success = False
    framedict_seqs = []
    steps_after_success = 0
    break_success_steps = 16
    step = 0
    while True:
        a = policy.get_action(obs)
        obs, reward, done, truncated, info = env.step(a)
        gripper = obs[3]  # 0 for fully close, 1 for fully open
        
        rendered_results = env.render()
        ee_pose = env.eef_pose

        if visualize:
            bgrs = []
            for name in env.CAMERA_NAMES:
                frame = rendered_results[name]
                if "corner" in name:
                    frame = rotate180(frame)
                
                bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
                bgr = draw_ee_proj(bgr, frame["K"], np.linalg.inv(frame["wcT"]), ee_pose)
                bgr = put_text(bgr, [name])
                bgrs.append(bgr)
            bgrs = np.concatenate(bgrs, axis=1)
            
            cv2.imshow("debug", bgrs)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        framedict = make_frame_dict(
            cam_frames=rendered_results,
            eef_pose=ee_pose,
            gripper=gripper,
            timestamp=step,
            gripper_action=a[-1],
        )
        framedict_seqs.append(framedict)

        if truncated:
            break
        
        if int(info["success"]) == 1:
            success = True
            steps_after_success += 1
        
        if steps_after_success > break_success_steps:
            break
        
        step += 1
        if step > max_ep_len:
            break

    env.close()
    return framedict_seqs, success


def make_dataset(framedict_seqs: list):
    data_dict = {
        "ee_pose": np.stack([f["ee_pose"] for f in framedict_seqs], axis=0).astype(np.float32),
        "gripper": np.array([f["gripper"] for f in framedict_seqs]).astype(np.float32),
        "gripper_desired": np.array([f["gripper_desired"] for f in framedict_seqs]).astype(np.float32),
        "timestamp": np.array([f["timestamp"] for f in framedict_seqs]).astype(np.float32)
    }
    
    for name in MyEnvWrapper.CAMERA_NAMES:
        data_dict[f"{name}/rgb"] = stack_or_compress(
            [f[name]["data"]["color"] for f in framedict_seqs],
            compress=True
        )
        
        data_dict[f"{name}/pose"] = np.stack(
            [f[name]["data"]["wcT"] for f in framedict_seqs],
            axis=0
        ).astype(np.float32)
        
        data_dict[f"{name}/K"] = np.stack(
            [f[name]["camera"]["K"] for f in framedict_seqs],
            axis=0
        ).astype(np.float32)
        
    return data_dict


task_description_mapping = {
    "Turn on faucet": "Rotate the faucet counter-clockwise.",
    "Sweep": "Sweep a puck off the table.",
    "Assemble nut": "Pick up a nut and place it onto a peg.",
    "Turn off faucet": "Rotate the faucet clockwise.",
    "Push": "Push the puck to a goal.",
    "Pull lever": "Pull a lever down 90 degrees.",
    "Turn dial": "Rotate a dial 180 degrees.",
    "Push with stick": "Grasp a stick and push a box using the stick.",
    "Get coffee": "Push a button on the coffee machine.",
    "Pull handle side": "Pull a handle up sideways.",
    "Basketball": "Dunk the basketball into the basket.",
    "Pull with stick": "Grasp a stick and pull a box with the stick.",
    "Sweep into hole": "Sweep a puck into a hole.",
    "Disassemble nut": "Pick a nut out of the a peg.",
    "Place onto shelf": "Pick and place a puck onto a shelf.",
    "Push mug": "Push a mug under a coffee machine.",
    "Press handle side": "Press a handle down sideways.",
    "Hammer": "Hammer a screw on the wall.",
    "Slide plate": "Slide a plate into a cabinet.",
    "Slide plate side": "Slide a plate into a cabinet sideways.",
    "Press button wall": "Bypass a wall and press a button.",
    "Press handle": "Press a handle down.",
    "Pull handle": "Pull a handle up.",
    "Soccer": "Kick a soccer into the goal.",
    "Retrieve plate side": "Get a plate from the cabinet sideways.",
    "Retrieve plate": "Get a plate from the cabinet.",
    "Close drawer": "Push and close a drawer.",
    "Press button top": "Press a button from the top.",
    "Reach": "Reach a goal position.",
    "Press button top wall": "Bypass a wall and press a button from the top.",
    "Reach with wall": "Bypass a wall and reach a goal.",
    "Insert peg side": "Insert a peg sideways.",
    "Pull": "Pull a puck to a goal.",#!!!!!!!!!!!!!!!!!!!!!!!!!!
    "Push with wall": "Bypass a wall and push a puck to a goal.",
    "Pick out of hole": "Pick up a puck from a hole.",
    "Pick&place w/ wall": "Pick a puck, bypass a wall and place the puck.",
    "Press button": "Press a button.",
    "Pick&place": "Pick and place a puck to a goal.",
    "Pull mug": "Pull a mug from a coffee machine.",
    "Unplug peg": "Unplug a peg sideways.",
    "Close window": "Push and close a window.",
    "Open window": "Push and open a window.",
    "Open door": "Open a door with a revolving joint.",
    "Close door": "Close a door with a revolving joint.",
    "Open drawer": "Open a drawer.",
    "Insert hand": "Insert the gripper into a hole.",
    "Close box": "Grasp the cover and close the box with it.",
    "Lock door": "Lock the door by rotating the lock clockwise.",
    "Unlock door": "Unlock the door by rotating the lock counter-clockwise.",
    "Pick bin": "Grasp the puck from one bin and place it into another bin."
}


name2short = {
    "assembly-v3": "Assemble nut",
    "basketball-v3": "Basketball",
    "bin-picking-v3": "Pick bin",
    "box-close-v3": "Close box",
    "button-press-topdown-v3": "Press button top",
    "button-press-topdown-wall-v3": "Press button top wall",
    "button-press-v3": "Press button",
    "button-press-wall-v3": "Press button wall",
    "coffee-button-v3": "Get coffee",
    "coffee-pull-v3": "Pull mug",
    "coffee-push-v3": "Push mug",
    "dial-turn-v3": "Turn dial",
    "disassemble-v3": "Disassemble nut",
    "door-close-v3": "Close door",
    "door-lock-v3": "Lock door",
    "door-open-v3": "Open door",
    "door-unlock-v3": "Unlock door",
    "drawer-close-v3": "Close drawer",
    "drawer-open-v3": "Open drawer",
    "faucet-close-v3": "Turn off faucet",
    "faucet-open-v3": "Turn on faucet",
    "hammer-v3": "Hammer",
    "hand-insert-v3": "Insert hand",
    "handle-press-side-v3": "Press handle side",
    "handle-press-v3": "Press handle",
    "handle-pull-v3": "Pull handle",
    "handle-pull-side-v3": "Pull handle side",
    "peg-insert-side-v3": "Insert peg side",
    "lever-pull-v3": "Pull lever",
    "peg-unplug-side-v3": "Unplug peg",
    "pick-out-of-hole-v3": "Pick out of hole",
    "pick-place-v3": "Pick&place",
    "pick-place-wall-v3": "Pick&place w/ wall",
    "plate-slide-back-side-v3": "Retrieve plate side",
    "plate-slide-back-v3": "Retrieve plate",
    "plate-slide-side-v3": "Slide plate side",
    "plate-slide-v3": "Slide plate",
    "reach-v3": "Reach",
    "reach-wall-v3": "Reach with wall",
    "push-back-v3": "Pull",
    "push-v3": "Push",
    "push-wall-v3": "Push with wall",
    "shelf-place-v3": "Place onto shelf",
    "soccer-v3": "Soccer",
    "stick-pull-v3": "Pull with stick",
    "stick-push-v3": "Push with stick",
    "sweep-into-v3": "Sweep into hole",
    "sweep-v3": "Sweep",
    "window-close-v3": "Close window",
    "window-open-v3": "Open window",
}



parser = argparse.ArgumentParser()
parser.add_argument("--output_root", type=str, default="./data_converted/metaworld")
parser.add_argument("--skip_saved", action="store_true", default=False)
parser.add_argument("--visualize", action="store_true", default=False)
opt = parser.parse_args()



mt50 = metaworld.MT50(seed=42)  # Construct the benchmark, sampling tasks
print([task.env_name for task in mt50.train_tasks])
print(len(mt50.train_tasks))


unique_task_names = list(mt50.train_classes.keys())

skip_saved = opt.skip_saved
save_data_root = opt.output_root
visualize = opt.visualize and ("DISPLAY" in os.environ)


for name_id, task_name in enumerate(tqdm(unique_task_names)):
    tasks = [task for task in mt50.train_tasks if task.env_name == task_name]
    prompt_text = task_description_mapping[name2short[task_name]]
    print("[INFO] task description = {}".format(prompt_text))

    for tid, task in enumerate(tqdm(tasks)):
        save_path = os.path.join(save_data_root, task_name, "{:0>4d}.h5".format(tid))
        if skip_saved and os.path.exists(save_path):
            print("[INFO] File {} already exists, skip.".format(save_path))
            continue

        env = mt50.train_classes[task_name]()
        env.set_task(task)
        env.reset()
        env = MyEnvWrapper(env)

        framedict_seqs, success = run_env(env, task_name)
        data_dict = make_dataset(framedict_seqs)

        # for k, v in data_dict.items():
        #     print("{}: {}".format(k, v.shape if isinstance(v, np.ndarray) else len(v)))
        # print(data_dict["gripper"])
        # print(data_dict["gripper_desired"])

        if success:
            attr_dict = {"compress": True, "prompt_text": prompt_text}
            write_to_h5(data_dict, attr_dict, save_path)

