import time
import torch
import numpy as np
from einops import rearrange
from typing import Union, Optional


Array = Union[np.ndarray, torch.Tensor]


class PinholeCamera(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        - width (int): The width in pixels of the camera.
        - height(int): The height in pixels of the camera.
        - K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def __str__(self):
        msg = "image (h, w) = {}".format(self.height, self.width)
        msg += "\nintrinsic = \n{}".format(self.K)
        return msg
    
    @property
    def K(self):
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]]
        )

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize intrinisic parameters from a dict object."""
        K = np.asarray(data["K"]).ravel()
        assert len(K) == 9
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=K[0],
            fy=K[4],
            cx=K[2],
            cy=K[5],
        )
        return intrinsic
    
    @classmethod
    def default(cls):
        return cls(width=640, height=480, fx=540, fy=540, cx=320, cy=240)

    def pixel_to_norm_camera_plane(self, uv: Array) -> Array:
        if isinstance(uv, np.ndarray):
            xy = (uv - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])
        elif isinstance(uv, torch.Tensor):
            xy = (uv - torch.tensor([self.cx, self.cy]).to(uv)) / \
                torch.tensor([self.fx, self.fy]).to(uv)
        return xy
    
    def norm_camera_plane_to_pixel(self, xy: Array, clip=True, round=False) -> Array:
        if isinstance(xy, np.ndarray):
            uv = xy * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])
            if clip: uv = np.clip(uv, 0, [self.width - 1, self.height - 1])
            if round: uv = np.round(uv).astype(np.int32)
        elif isinstance(xy, torch.Tensor):
            uv = xy * torch.tensor([self.fx, self.fy]).to(xy) + \
                torch.tensor([self.cx, self.cy]).to(xy)
            if clip:
                uv[..., 0].clip_(0, self.width - 1)
                uv[..., 1].clip_(0, self.height - 1)
            if round: uv = torch.round(uv).to(torch.int32)
        return uv
    
    def project(self, points: Array, wcT: Optional[Array] = None, to_pix=False) -> Array:
        """Project 3d points into pixel coordinate system or 
        normalized point coordinate system

        Arguments:
        - points: (N, 3)
        - wcT: camera extrinsic, (4, 4)

        Returns:
        - uv: (N, 2), pixel coordinates if to_pix is True, 
            otherwise normalized camera plane coordinates
        """
        if wcT is not None:
            cwT = np.linalg.inv(wcT) if isinstance(wcT, np.ndarray) else \
                torch.inverse(wcT)
            points = points @ cwT[:3, :3].T + cwT[:3, 3]
        
        xy = points[..., :2] / (points[..., 2:3] + 1e-16)
        if to_pix:
            return self.norm_camera_plane_to_pixel(xy, clip=False, round=False)
        else:
            return xy
    
    def inverse_project(self, uv: Array, Z: Array, wcT: Optional[Array] = None) -> Array:
        """Inverse projection pixel coordinates to 3D points

        Arguments:
        - uv: pixel coordinates, (N, 2),
        - Z: depth, (N)
        - wcT: camera extrinsic, (4, 4)

        Returns:
        - XYZ: points in 3D space
        """
        xy = self.pixel_to_norm_camera_plane(uv)
        XY = xy * Z[..., None]
        cat_fn = np.concatenate if isinstance(uv, np.ndarray) else torch.cat
        XYZ = cat_fn([XY, Z[..., None]], axis=-1)
        if wcT is not None:
            XYZ = XYZ @ wcT[:3, :3].T + wcT[:3, 3]
        return XYZ


class OpenglCamera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, intrinsic: PinholeCamera, near=0.01, far=4):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.gl_proj_matrix = self.proj_matrix.flatten(order="F")

    def to_dict(self):
        data = {
            "intrinsic": self.intrinsic.to_dict(),
            "near": self.near, 
            "far": self.far, 
        }
        return data

    @classmethod
    def from_dict(cls, data: dict):
        intrinsic = PinholeCamera.from_dict(data["intrinsic"])
        near = data["near"]
        far = data["far"]
        return cls(intrinsic, near, far)

    def to_isaac(self, pixel_size=3e-3):
        return dict(
            resolution = (self.intrinsic.width, self.intrinsic.height),
            horizontal_aperture = pixel_size * self.intrinsic.width,
            vertical_aperture = pixel_size * self.intrinsic.height,
            focal_length_x = self.intrinsic.fx * pixel_size,
            focal_length_y = self.intrinsic.fy * pixel_size,
            clipping_range = (self.near, self.far)
        )

    def render_bullet(self, wcT: Optional[np.ndarray], client=0):
        """Render synthetic RGB and depth images.

        Arguments:
        - wcT: camera extrinsic, (^{world}_{cam} T).
        """
        # Construct OpenGL compatible view and projection matrices.
        cwT = np.linalg.inv(wcT) if wcT is not None else np.eye(4)
        gl_view_matrix = cwT
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")

        import pybullet as p
        result = p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=self.gl_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client
        )

        rgb = np.ascontiguousarray(result[2][:, :, :3])
        z_buffer = result[3]
        seg = result[4]

        depth = self.far * self.near / (
            self.far - (self.far - self.near) * z_buffer)

        return Frame(self, rgb, depth, seg, wcT)

    def project(self, points: Array, wcT: Optional[Array] = None) -> Array:
        """Project 3d points to pixel coordinates

        Arguments:
        - points: (N, 3), points in world frame
        - wcT: (4, 4), wcT

        Returns:
        - uv: (N, 2), pixel coordinates.
            If points are calculated from the the same frame, then: 
                range of u: [0, W-1];
                range of v: [0, H-1];
        """
        backend = np if isinstance(points, np.ndarray) else torch
        use_numpy = isinstance(points, np.ndarray)

        W, H = self.intrinsic.width, self.intrinsic.height
        N = len(points)
        ones = np.ones((N, 1), points.dtype) if use_numpy else \
            torch.ones((N, 1)).to(points)
        points_homo = backend.concatenate([points, ones], axis=-1)

        if wcT is not None:
            cwT = backend.linalg.inv(wcT)
            proj_mat = self.proj_matrix if use_numpy else \
                torch.from_numpy(self.proj_matrix).to(points)
            proj = proj_mat @ cwT @ points_homo.T  # (4, N)
        else:
            proj = points_homo.T
        proj = (proj / proj[-1, :]).T  # (N, 4)
        proj[:, 0] = -proj[:, 0]
        WH = np.array([W, H]) if use_numpy else torch.tensor([W, H]).to(points)
        uv = (proj[:, :2] + 1.) * WH / 2.
        return uv
    
    def inverse_project(self, uv: Array, Z: Array, wcT: Optional[Array] = None) -> Array:
        """Inverse projection, get corresponding 3d positions of pixel coordinates

        Arguments:
        - uv: (N, 2), pixel coordinates,
                range of u: [0, W-1];
                range of v: [0, H-1];
        - Z: (N,), depth in camera frame
        - wcT: (4, 4), ^{world} _{cam} T
        
        Returns:
        - points: (N, 3)
        """
        backend = np if isinstance(uv, np.ndarray) else torch
        use_numpy = isinstance(uv, np.ndarray)

        W, H = self.intrinsic.width, self.intrinsic.height
        N = len(uv)

        WH = np.array([W, H]) if use_numpy else torch.tensor([W, H]).to(uv)
        inv_proj = uv * 2. / WH - 1.
        inv_proj[:, 0] = -inv_proj[:, 0]

        f, n = self.far, self.near
        norm_Z = (f+n)/(f-n) + 2*n*f/(f-n) * 1./Z

        ones = np.ones((N, 1), uv.dtype) if use_numpy else torch.ones((N, 1)).to(uv)
        inv_proj = backend.concatenate([inv_proj, norm_Z[..., None], ones], axis=-1)
        inv_proj = inv_proj * -Z[:, None]  # (N, 4)

        X = (inv_proj[:, 0] - Z*self.proj_matrix[0, 2]) / self.proj_matrix[0, 0]
        Y = (inv_proj[:, 1] - Z*self.proj_matrix[1, 2]) / self.proj_matrix[1, 1]
        points = backend.stack([X, Y, Z], axis=-1)  # (N, 3)
        if wcT is not None:
            points = points @ wcT[:3, :3].T + wcT[:3, 3]

        # # equal implementation
        # inv_proj = np.linalg.inv(self.proj_matrix @ cwT) @ inv_proj.T  # (4, N)
        # points = inv_proj[:3, :].T
        return points


class Frame(object):
    def __init__(
        self, 
        camera: Union[PinholeCamera, OpenglCamera], 
        color: Array, 
        depth: Optional[Array] = None, 
        seg: Optional[dict] = None, 
        wcT: Optional[Array] = None,
        pc_camera: Optional[Array] = None,
        pc_world: Optional[Array] = None,
        timestep: Optional[float] = None,
        encoding: str = "rgb"
    ):
        """
        Arguments:
        - camera: Camera instance
        - color: (H, W, 3 or 4) for rgb or rgba image
        - depth: (H, W, [1]) for depth image
        - seg: segmentation information
        - wcT: camera extrinsic, ^{world}_{cam} T, 4x4 transformation matrix
        """
        if (depth is not None) and (depth.ndim == 3):
            assert depth.shape[-1] == 1
            depth = depth.squeeze(-1)

        self.encoding = encoding
        self.camera = camera
        self.color = color
        self.depth = depth
        self.seg = seg
        self.wcT = wcT

        if isinstance(color, np.ndarray) or isinstance(depth, np.ndarray):
            # self.backend = np  # module cannot be pickled
            self.use_numpy = True
        else:
            # self.backend = torch  # module cannot be pickled
            self.use_numpy = False

        # cache
        self._pc_camera = pc_camera
        self._pc_world = pc_world

        if timestep is None:
            timestep = time.perf_counter()
        self.timestep = timestep
    
    def to_dict(self):
        return {
            "model": "pinhole" if isinstance(self.camera, PinholeCamera) else 
                     "opengl",
            "camera": self.camera.to_dict(),
            "data": {
                "color": self.color,
                "depth": self.depth,
                "seg": self.seg,
                "wcT": self.wcT,
                "pc_camera": self._pc_camera,
                "pc_world": self._pc_world,
                "timestep": self.timestep,
                "encoding": self.encoding
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        try:
            cam_cls = PinholeCamera if data["model"] == "pinhole" else OpenglCamera
            camera = cam_cls.from_dict(data["camera"])
            return cls(camera, **data["data"])
        except Exception as e:
            print(e)
            return cls(**data)
    
    def _pointcloud(self, wcT: Optional[Array]):
        backend = np if self.use_numpy else torch

        H, W = self.depth.shape[:2]
        xv, yv = backend.meshgrid(
            backend.arange(W), 
            backend.arange(H), 
            indexing="xy"
        )
        pixel_coords = backend.stack([xv, yv], axis=-1)
        if not self.use_numpy:
            pixel_coords = pixel_coords.to(self.depth)
        pixel_coords = rearrange(pixel_coords, "h w c -> (h w) c")
        points3d = self.camera.inverse_project(
            pixel_coords, self.depth.ravel(), wcT)
        points3d = rearrange(points3d, "(h w) c -> h w c", h=H, w=W)
        return points3d

    @property
    def pc_camera(self):
        """Point cloud in camera frame, returns (H, W, 3)"""
        if self._pc_camera is None:
            self._pc_camera = self._pointcloud(None)
        return self._pc_camera
    
    @property
    def pc_world(self):
        """Point cloud in world frame, returns (H, W, 3)"""
        if self._pc_world is None:
            self._pc_world = self._pointcloud(self.wcT)
        return self._pc_world
    
    def semantic_mask(self, valid_label_subnames = []):
        if self.seg is None:
            H, W = self.color.shape[:2]
            mask = np.ones((H, W), dtype=bool)
            return mask

        if isinstance(self.seg, np.ndarray):
            return self.seg

        seg_data: np.ndarray = self.seg["data"]
        seg_info: dict = self.seg["info"]
        mask = np.zeros(seg_data.shape, dtype=bool)

        for label, name in seg_info["idToLabels"].items():
            for query_name in valid_label_subnames:
                if query_name in name["class"]:
                    mask[seg_data == int(label)] = True
                    break
        return mask


def _build_projection_matrix(intrinsic: PinholeCamera, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho


def look_at_view_rotation(
    eye: Array,
    to: Array,
    up: Array
) -> Array:
    backend = np if isinstance(eye, np.ndarray) else torch

    z = to - eye
    z = z / (backend.linalg.norm(z, axis=-1)[..., None] + 1e-15)

    y = -up / (backend.linalg.norm(up, axis=-1)[..., None] + 1e-15)
    x = backend.cross(y, z, axis=-1)
    x = x / (backend.linalg.norm(x, axis=-1)[..., None] + 1e-15)

    y = backend.cross(z, x, axis=-1)
    return backend.stack([x, y, z], axis=-1)


def look_at_view_transform(
    eye: Array,
    to: Array,
    up: Array
):
    T_shape = list(eye.shape) + [4]
    T_shape[-2] = 4

    R = look_at_view_rotation(eye, to, up)

    if isinstance(R, np.ndarray):
        T = np.zeros(T_shape, dtype=R.dtype)
    else:
        T = R.new_zeros(T_shape)
    
    T[..., :3, :3] = look_at_view_rotation(eye, to, up)
    T[..., :3, 3] = eye
    T[..., 3, 3] = 1.0
    return T


if __name__ == "__main__":
    T = look_at_view_transform(np.array([1, 1, 1]), 
                               np.array([0, 0, 0]), 
                               np.array([0, 0, 1]))
    print(T)
