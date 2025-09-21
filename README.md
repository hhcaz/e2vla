# e2vla

Codes under refactoring


# Dependency


# Pretrain
## 1. Dataset Preparation
* Droid
  
  We use the processed data from [cadence/droid_1.0.1](https://huggingface.co/datasets/cadene/droid_1.0.1) as it has camera extrinsic attached. Download it to anywhere you like, and make a symbolic link to it as `./data_raw/droid_1.0.1`. Then run:
  ```bash
  conda activate lerobot
  python data_prepare/process_droid.py \
    --input_root ./data_raw/droid_1.0.1 \
    --alter_vid_root VIDEO_DOWNLOAD_PATH \
    --output_root ./data_converted/droid \
    --skip_saved
  ```
  **Note:**
  * This requires [lerobot](https://github.com/huggingface/lerobot) installed. We use version 0.1.0. You may need to create a new conda environment (e.g. named `lerobot`) and install the package via:
    ```bash
    pip install "lerobot==0.1.0"
    ```
  * The initial downloads of video files may be incomplete (test at 2025/04). We need to download the full video files and place them at `VIDEO_DOWNLOAD_PATH`. TODO: upload scripts to fix this.
  
* Maniskill
  
  First download the [data](https://www.tensorflow.org/datasets/catalog/maniskill_dataset_converted_externally_to_rlds) to anywhere you like, e.g.:
  ```bash
  mkdir -p ANYWHERE/maniskill
  gsutil -m cp -r gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0 ANYWHERE/maniskill
  ln -s ANYWHERE/maniskill ./data_raw/maniskill
  ```
  Then run:
  ```bash
  conda activate tensorflow
  python data_prepare/process_maniskill.py \
    --input_root ./data_raw/maniskill/0.1.0 \
    --output_root ./data_converted/maniskill/0.1.0 \
    --visualize
  ```
  Note:
  * This requires [tensorflow](https://www.tensorflow.org/install) installed. You may need to create a new conda environment (e.g. named `tensorflow`) to install it and run the above command to generate data.
  
* Metaworld
  
  This doesn't require downloading extra data. However, you may still need to create a new conda environment (e.g. named `metaworld-v3`) and then install the [metaworld](https://github.com/Farama-Foundation/Metaworld) package via:
  ```bash
  pip install "metaworld==2.0.0"
  ```
  Then run:
  ```bash
  conda activate metaworld-v3
  python data_prepare/process_metaworld.py \
    --output_root ./data_converted/metaworld \
    --visualize \
    --skip_saved
  ```
  Note:
  * Although we install "metaworld==2.0.0", it is actually version 3.

If you have downloaded and processed all the data, the file structure would be like this: 
TODO: add an image


## 2. Run training
You can use `python train.py -h` to see the help message. To pretrain on the above three datasets, run:
```
CUDA_VISIBLE_DEVICES=x python train.py --config pretrain -s EXPERIMENT_NAME
```


