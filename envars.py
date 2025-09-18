import os
import cv2


# import torchvision时会顺带着import av
# 但是av>=9会与opencv冲突导致cv2.imshow卡死
# 所以要先import cv2并调用cv2.namedWindow，再import torchvision
if os.environ.get("DISPLAY"):
    # https://github.com/pytorch/vision/issues/5940
    # https://github.com/huggingface/lerobot/pull/757
    try:
        cv2.namedWindow("_show_before_importing_av")
        cv2.destroyWindow("_show_before_importing_av")
    except Exception:
        pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
