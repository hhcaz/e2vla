eval "$(conda shell.bash hook)"
conda activate libero


# first download dataset from:
# https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets


python data_prepare/process_libero.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir ./data_raw/libero \
    --libero_target_dir ./data_converted/libero \
    --skip_saved \
    --visualize


python data_prepare/process_libero.py \
    --libero_task_suite libero_object \
    --libero_raw_data_dir ./data_raw/libero \
    --libero_target_dir ./data_converted/libero \
    --skip_saved \
    --visualize


python data_prepare/process_libero.py \
    --libero_task_suite libero_goal \
    --libero_raw_data_dir ./data_raw/libero \
    --libero_target_dir ./data_converted/libero \
    --skip_saved \
    --visualize


python data_prepare/process_libero.py \
    --libero_task_suite libero_10 \
    --libero_raw_data_dir ./data_raw/libero \
    --libero_target_dir ./data_converted/libero \
    --skip_saved \
    --visualize


# # libero_90 is not used
# python data_prepare/process_libero.py \
#     --libero_task_suite libero_90 \
#     --libero_raw_data_dir ./data_raw/libero \
#     --libero_target_dir ./data_converted/libero \
#     --skip_saved \
#     --visualize
