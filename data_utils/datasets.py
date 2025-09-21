import os
import sys
import glob
import random
import inspect
import traceback
from typing import Dict
from data_utils.dataset_base import DataConfig, H5DatasetMapBase


# __getitem__(self, i) of H5DatasetMapBase return a dict of Tensors (except for `prompt_text`),
# We list the name and shape as follows:

# K:                    (ncam, 3, 3)
# obs_rgbs:             (To, ncam, 3, H, W)
# obs_masks:            (To, ncam, H, W)
# prompt_text:          str
# obs_norm_xys:         (To, ncam, 2, H, W), coordinates in normalized camera plane, 2 = (x, y)
# obs_extrinsics:       (To, ncam, 4, 4)
# current_ee_pose:      (nee, 4, 4)
# history_ee_states:    (nhist, nee, 17), 17 = (16 for flattened 4x4 pose matrix (row major), 1 for gripper)
# gt_future_ee_states:  (Ta, nee, 17)
# timestamps:           (To,)
# valid_ee_mask:        (nee,)

# NOTE:
# To means number of image observations
# ncam means number of cameras
# nee means number of end-effectors
# nhist means number of historical actions (including current, therefore nhist always >= 1)
# Ta means number of feature actions


class Libero(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("agentview", "eye_in_hand"),
    )

    @classmethod
    def inst(cls, task_suites = []):
        if isinstance(task_suites, str):
            task_suites = [task_suites]
        
        h5_files = []
        for suite in task_suites:
            if suite == "spatial":
                h5_files.extend(glob.glob("./data_converted/libero/libero_spatial_no_noops/**/*.h5", recursive=True))
            elif suite == "object":
                h5_files.extend(glob.glob("./data_converted/libero/libero_object_no_noops/**/*.h5", recursive=True))
            elif suite == "goal":
                h5_files.extend(glob.glob("./data_converted/libero/libero_goal_no_noops/**/*.h5", recursive=True))
            elif suite == "90":
                h5_files.extend(glob.glob("./data_converted/libero/libero_90_no_noops/**/*.h5", recursive=True))
            elif suite == "10":
                h5_files.extend(glob.glob("./data_converted/libero/libero_10_no_noops/**/*.h5", recursive=True))
            else:
                raise TypeError("Unknown task suite: {}".format(suite))
        
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)
    
    def modify_prompt(self, lang: str):
        # remove something like "LIVING ROOM SCENE6" in libero10 and libero90
        index = lang.find("SCENE")
        if index >= 0:
            lang = lang[index:]
            lang = " ".join(lang.split(" ")[1:])
        return lang
    
    def __getitem__(self, i):
        out = super().__getitem__(i)
        out["prompt_text"] = self.modify_prompt(out["prompt_text"])
        return out


class LiberoSpatial(Libero):
    @classmethod
    def inst(cls):
        return super().inst(["spatial"])


class LiberoObject(Libero):
    @classmethod
    def inst(cls):
        return super().inst(["object"])


class LiberoGoal(Libero):
    @classmethod
    def inst(cls):
        return super().inst(["goal"])


class Libero10(Libero):
    @classmethod
    def inst(cls):
        return super().inst(["10"])


class Libero90(Libero):
    @classmethod
    def inst(cls):
        return super().inst(["90"])


class Maniskill(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("main_camera", "wrist_camera"),
    )

    @classmethod
    def inst(cls):
        h5_files = glob.glob("./data_converted/maniskill/0.1.0/*.h5")
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)


class MetaWorld(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("corner", "topview", "gripperPOV"),
    )

    @classmethod
    def inst(cls):
        h5_files = glob.glob("./data_converted/metaworld/*/*.h5")
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)


class Droid(H5DatasetMapBase):
    config = DataConfig(
        record_dt=1.0/15,
        sample_dt=1.0/15,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("exterior_2_left", "exterior_1_left", "wrist_left"),
        sample_state_gaps=2,
        video_root="./data_converted/droid"
    )

    @classmethod
    def filter_files(cls, filelist):
        filtered = []
        for f in filelist:
            # remove .h5 and episode_
            episode_index = int(os.path.split(f)[-1][:-3].replace("episode_", ""))
            if episode_index in [11907, 14419, 24440, 64837, 64871]:
                print("[INFO] in Droid dataset, remove file: {}".format(f))
            else:
                filtered.append(f)
        return filtered

    @classmethod
    def inst(cls):
        h5_files = glob.glob("./data_converted/droid/data/*/*.h5", recursive=True)
        h5_files = cls.filter_files(h5_files)
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)
    
    def adjust_ee_pose(self, out):
        fwd_axis = 2  # zaxis
        gripper_length = 0.15
        for key in ["current_ee_pose"]:
            out[key][..., :3, 3] += out[key][..., :3, fwd_axis] * gripper_length
        for key in ["history_ee_states", "gt_future_ee_states"]:
            ee = out[key]  # (B, T, Nee, 17)
            Ta, Nee, _ = ee.shape
            pose = ee[..., :16].reshape(Ta, Nee, 4, 4)
            pose[..., :3, 3] += pose[..., :3, fwd_axis] * gripper_length
            out[key][..., :16] = pose.reshape(Ta, Nee, 16)
        return out
    
    def __getitem__(self, i):
        try:
            out = super().__getitem__(i)
        except Exception as e:
            # This occasionally fails, I don't know why
            print("[INFO] Error when getitem {}".format(i))
            traceback.print_exc()
            print("[INFO] Retry loading another index")
            out = super().__getitem__((i+1)%len(self))
            
        out = self.adjust_ee_pose(out)
        return out


class PickPlaceCan(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("e2h_cam", "eih_cam")
    )

    def enrich_prompt(self, lang: str):
        prompts = {
            "7up01": ["green can", "green bottle", "green soda"],
            "Coke01": ["red can", "red bottle", "red soda", "red Coca Cola"],
            "Pepsi01": ["blue can", "blue bottle", "blue soda", "blue Pepsi"],
            "Pepsi02": ["black can", "black bottle", "black soda", "black Pepsi"]
        }
        return random.sample(prompts[lang], k=1)[0]
    
    def __getitem__(self, i):
        out = super().__getitem__(i)
        template = "move the {} to the plate"
        out["prompt_text"] = template.format(self.enrich_prompt(out["prompt_text"]))
        return out
    
    @classmethod
    def inst(cls):
        h5_files = glob.glob("./data_converted/pick-place-can/*.h5")
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)


class OpenDrawer(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("agent_camera",)
    )

    @classmethod
    def inst(cls):
        h5_files = glob.glob("./data_converted/drawer/*.h5")
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)


class OpenOven(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("agent_camera",)
    )

    @classmethod
    def inst(cls):
        h5_files = glob.glob("./data_converted/oven/*.h5")
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)


def get_subclasses(base_class):
    current_module = sys.modules[__name__]
    subclasses = []
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if issubclass(obj, base_class) and obj is not base_class:
            subclasses.append(obj)
    return subclasses


DATA_CONFIGS: Dict[str, DataConfig] = {
    c.__name__: c.config for c in get_subclasses(H5DatasetMapBase)
}

