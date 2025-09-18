eval "$(conda shell.bash hook)"
conda activate lerobot


# first download dataset from:
# https://huggingface.co/datasets/cadene/droid_1.0.1


python data_prepare/process_droid.py \
    --input_root ./data_raw/droid_1.0.1 \
    --alter_vid_root /nas_data_new/caz/aria2_downloads \
    --output_root ./data_converted/droid \
    --skip_saved
