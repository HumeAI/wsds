#!/bin/bash
#SBATCH --job-name=dataset_conversion
#SBATCH --output=/mnt/weka/hoon/wsds/logs/dataset_conversion_%A_%a.out
#SBATCH --error=/mnt/weka/hoon/wsds/logs/dataset_conversion_%A_%a.err
#SBATCH --time=192:00:00
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --array=0-4

# Create logs directory if it doesn't exist
mkdir -p /mnt/weka/hoon/wsds/logs

# Load conda
source /mnt/weka/hoon/miniconda3/etc/profile.d/conda.sh
conda activate hume_wsds

# Define the conversion script path
CONVERT_SCRIPT="/mnt/weka/hoon/wsds/examples/dataset_conversion/convert_dataset.sh"

# Array of YAML config files to process (one per array task)
CONFIG_FILES=(
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_1M_ar-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_de-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_en-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_fr-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_ko-vad_ws_continuous.yaml"
    "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_pl-vad_ws_continuous.yaml"
    "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_pt-vad_ws_continuous.yaml"
    "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_ru-vad_ws_continuous.yaml"
    "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v3_de-vad_ws_continuous.yaml"
    "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v3_en-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v4_multilingual-vad_ws.yaml"
)

# Validate array index
IDX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}
if (( IDX < 0 || IDX >= ${#CONFIG_FILES[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range (0..$((${#CONFIG_FILES[@]}-1)))"
  exit 1
fi

config_file="${CONFIG_FILES[$IDX]}"
config_name=$(basename "$config_file" .yaml)

echo "Starting dataset conversion task"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Config index: $IDX"
echo "Config file: $config_file"
echo "Start time: $(date)"
echo "=========================================="

if [[ ! -f "$config_file" ]]; then
  echo "ERROR: Config file not found: $config_file"
  exit 1
fi

# Run conversion
if bash "$CONVERT_SCRIPT" "$config_file"; then
  echo "✅ SUCCESS: $config_name"
else
  echo "❌ FAILED:  $config_name"
  exit 1
fi

echo "End time: $(date)"
echo "Task complete."
