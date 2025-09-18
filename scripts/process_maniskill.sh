eval "$(conda shell.bash hook)"
conda activate lerobot

# first download data (~150G):
# mkdir -p ./data_raw/maniskill
# gsutil -m cp -r gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0 ./data_raw/maniskill


python data_prepare/process_maniskill.py \
    --input_root ./data_raw/maniskill/0.1.0 \
    --output_root ./data_converted/maniskill/0.1.0 \
    --visualize
