import os
import cv2
import numpy as np


class VideoWriter(object):
    def __init__(self, output_path: str, frame_rate: float):
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = None
        self.finalized = False
    
    def _lazy_init(self, bgr: np.ndarray):
        H, W = bgr.shape[:2]
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.writer = cv2.VideoWriter(self.output_path, self.fourcc, 
                                      self.frame_rate, (W, H))
    
    def write(self, bgr: np.ndarray):
        assert not self.finalized, "cannot call `write` after `finalize` is called"
        if self.writer is None:
            self._lazy_init(bgr)
        self.writer.write(bgr)
    
    def finalize(self):
        if self.writer is not None:
            print("[INFO] File save to {}".format(self.output_path))
            self.writer.release()
            self.writer = None
        self.finalized = True

        # rewrite with ffmpeg
        tmp_output = self.output_path.replace(".mp4", "_tmp.mp4")
        os.rename(self.output_path, tmp_output)
        os.system(
            f"ffmpeg -y -i {tmp_output} -c:v libx264 -preset fast -crf 23 "
            f"-pix_fmt yuv420p {self.output_path}"
        )
        os.remove(tmp_output)

