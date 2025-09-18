import numpy as np
import open3d as o3d


class Viewer(object):
    def __init__(self, default_cam_pose: np.ndarray = np.eye(4)):
        self.pcd = None
        self.axes = None
        self.default_cam_pose = default_cam_pose
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=640, height=480)
        # https://www.glfw.org/docs/latest/group__keys.html
        self.vis.register_key_callback(67, self.reset_cam_pose)  #define GLFW_KEY_C 67
    
    def reset_cam_pose(self, vis):
        view_ctl = self.vis.get_view_control()
        params = view_ctl.convert_to_pinhole_camera_parameters()
        params.extrinsic = self.default_cam_pose
        view_ctl.convert_from_pinhole_camera_parameters(params, True)
    
    def update_pcd(self, rgb: np.ndarray, xyz: np.ndarray):
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.
        
        rgb = np.reshape(rgb, (-1, 3)).astype(np.float32)
        xyz = np.reshape(xyz, (-1, 3)).astype(np.float32)
        
        if self.pcd is None:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(xyz)
            self.pcd.colors = o3d.utility.Vector3dVector(rgb)
            self.vis.add_geometry(self.pcd)
            self.reset_cam_pose(self.vis)
        else:
            xyz_interface = np.asarray(self.pcd.points)
            rgb_interface = np.asarray(self.pcd.colors)
            xyz_interface[:] = xyz
            rgb_interface[:] = rgb
        return self
    
    def update_axes(self, poses: np.ndarray):
        """
        Args:
            poses (np.ndarray): shape (N, 4, 4)
        """
        if self.axes is None:
            self.axes = []
            for pose in poses:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
                axis.scale(0.05, center=np.zeros(3))
                axis.transform(pose)
                self.axes.append(axis)
            for axis in self.axes:
                self.vis.add_geometry(axis)
            self.prev_axes_poses = poses.copy()
            self.reset_cam_pose(self.vis)
        else:
            for i, axis in enumerate(self.axes):
                delta_transform = poses[i] @ np.linalg.inv(self.prev_axes_poses[i])
                axis.transform(delta_transform)
            self.prev_axes_poses = poses.copy()
        return self
    
    def render(self):
        if self.pcd is not None:
            self.vis.update_geometry(self.pcd)
        if self.axes is not None:
            for axis in self.axes:
                self.vis.update_geometry(axis)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def screenshot(self, do_render: bool = True):
        image = self.vis.capture_screen_float_buffer(do_render=do_render)
        image = np.asarray(image)
        return image
