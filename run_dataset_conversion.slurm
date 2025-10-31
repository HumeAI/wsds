#!/bin/bash
#SBATCH --job-name=dataset_conversion
#SBATCH --output=/mnt/weka/hoon/wsds/logs/dataset_conversion_%j.out
#SBATCH --error=/mnt/weka/hoon/wsds/logs/dataset_conversion_%j.err
#SBATCH --time=192:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=64G
#SBATCH --partition=train
#SBATCH --gres=gpu:8

# Create logs directory if it doesn't exist
mkdir -p /mnt/weka/hoon/wsds/logs

# Load conda
source /mnt/weka/hoon/miniconda3/etc/profile.d/conda.sh
conda activate hume_wsds

# Define the conversion script path
CONVERT_SCRIPT="/mnt/weka/hoon/wsds/examples/dataset_conversion/convert_dataset.sh"

# Array of YAML config files to process
CONFIG_FILES=(
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_1M_ar-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_de-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_en-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_fr-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_ko-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_pl-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_pt-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v2_ru-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v3_de-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v3_en-vad_ws_continuous.yaml"
    # "/mnt/weka/hoon/wsds/examples/dataset_conversion/configs/wyndlabs_v4_multilingual-vad_ws.yaml"
)

# Track success/failure counts
success_count=0
failure_count=0
failed_configs=()

echo "Starting dataset conversion batch job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Total configs: ${#CONFIG_FILES[@]}"
echo ""

# Process each config file
for config_file in "${CONFIG_FILES[@]}"; do
    config_name=$(basename "$config_file" .yaml)

    echo "=========================================="
    echo "Processing: $config_name"
    echo "Time: $(date)"
    echo "=========================================="

    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file not found: $config_file"
        ((failure_count++))
        failed_configs+=("$config_name")
        continue
    fi

    # Run conversion and capture exit code
    if bash "$CONVERT_SCRIPT" "$config_file"; then
        echo "✅ SUCCESS: $config_name"
        ((success_count++))
    else
        echo "❌ FAILED: $config_name"
        ((failure_count++))
        failed_configs+=("$config_name")
    fi

    echo ""
done

# Final summary
echo "=========================================="
echo "BATCH CONVERSION SUMMARY"
echo "=========================================="
echo "End time: $(date)"
echo "Total configs: ${#CONFIG_FILES[@]}"
echo "Successful: $success_count"
echo "Failed: $failure_count"

if [[ $failure_count -gt 0 ]]; then
    echo ""
    echo "Failed configs:"
    for failed_config in "${failed_configs[@]}"; do
        echo "  - $failed_config"
    done
fi

echo ""
echo "Job complete!"
