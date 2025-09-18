eval "$(conda shell.bash hook)"
conda activate metaworld-v3


python data_prepare/process_metaworld.py \
    --output_root ./data_converted/metaworld \
    --visualize \
    --skip_saved


