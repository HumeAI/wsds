#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-config.yaml}"

to_bool() {
  [[ "${1:-}" == "true" ]] && echo "true" || echo "false"
}

CONVERT_AUDIO=$(to_bool "$(yq -r '.flags.convert_audio' "$CFG")")
CONVERT_MVAD=$(to_bool "$(yq -r '.flags.convert_mvad' "$CFG")")
CONVERT_ARTIFACTS=$(to_bool "$(yq -r '.flags.convert_artifacts' "$CFG")")
CREATE_INDEX=$(to_bool "$(yq -r '.flags.create_index' "$CFG")")
MVAD_DIR_NAME=$(yq -r '.flags.mvad_dir_name // "mvad"' "$CFG")

AUDIO_BASE=$(yq -r '.paths.audio_base' "$CFG")
INPUT_BASE=$(yq -r '.paths.input_base' "$CFG")
OUTPUT_BASE=$(yq -r '.paths.output_base' "$CFG")

SEGMENTATION_TYPE=$(yq -r '.segmentation_type' "$CFG")

mapfile -t SUBDIRS < <(yq -r '.subdirs[]' "$CFG")

MVAD_DIR="${SEGMENTATION_TYPE}_mvad"

mkdir -p "$OUTPUT_BASE/source/audio" "$OUTPUT_BASE/source/$MVAD_DIR" "$OUTPUT_BASE/$SEGMENTATION_TYPE"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; BOLD='\033[1m'; RST='\033[0m'

declare -A counts
total=0

### audio ###
if [[ "$CONVERT_AUDIO" == "true" ]]; then
  echo -e "${BLU}${BOLD}=== Converting audio ===${RST}"
  parallel --plus --tag --bar \
    wsds shard_from_webdataset {} "$OUTPUT_BASE/source/audio/{/.}.wsds" \
    ::: "$AUDIO_BASE"/*.tar

  # Count .wsds files before renaming (optional)
  counts["source/audio"]=$(ls -1 "$OUTPUT_BASE/source/audio"/*.wsds 2>/dev/null | wc -l || echo 0)

  # Rename .source_separation.wsds → .wsds
  for f in "$OUTPUT_BASE/source/audio"/*.source_separation.wsds; do
    new_name="${f/source_separation./}"
    if [[ "$f" != "$new_name" ]]; then
      mv "$f" "$new_name"
    fi
  done
fi

### mvad ###
if [[ "$CONVERT_MVAD" == "true" ]]; then
  echo -e "${BLU}${BOLD}=== Converting $MVAD_DIR ===${RST}"
  parallel --plus --tag --bar \
    wsds shard_from_webdataset {} "$OUTPUT_BASE/source/$MVAD_DIR/{/...}.wsds" \
    ::: "$INPUT_BASE/$MVAD_DIR_NAME"/*.tar.gz
  counts["source/$MVAD_DIR"]=$(ls -1 "$OUTPUT_BASE/source/$MVAD_DIR"/*.wsds 2>/dev/null | wc -l || echo 0)
fi

### artifacts ###
if [[ "$CONVERT_ARTIFACTS" == "true" ]]; then
  for sub in "${SUBDIRS[@]}"; do
    in_dir="$INPUT_BASE/$sub"
    out_dir="$OUTPUT_BASE/$SEGMENTATION_TYPE/$sub"
    if [[ ! -d "$in_dir" ]]; then
      echo -e "${YLW}Skip missing input dir:${RST} $in_dir"
      continue
    fi
    mkdir -p "$out_dir"
    echo -e "${BLU}${BOLD}=== Converting $sub ===${RST}"
    parallel --plus --tag --bar \
      wsds shard_from_webdataset {} "$out_dir/{/...}.wsds" \
      --compression no-compression \
      ::: "$in_dir"/*.tar.gz
    counts["$SEGMENTATION_TYPE/$sub"]=$(ls -1 "$out_dir"/*.wsds 2>/dev/null | wc -l || echo 0)
  done
fi

### index creation ###
if [[ "$CREATE_INDEX" == "true" ]]; then
  echo -e "${BLU}${BOLD}=== Creating indexes ===${RST}"
  rm -f "$OUTPUT_BASE/source/index.sqlite3"
  rm -f "$OUTPUT_BASE/$SEGMENTATION_TYPE/index.sqlite3"
  rm -f "$OUTPUT_BASE/$SEGMENTATION_TYPE/audio.wsds-link"

  echo "wsds init $OUTPUT_BASE/source"
  wsds init "$OUTPUT_BASE/source"

  VAD_COLUMN="${MVAD_DIR}.raw.vad.npy"
  if [[ "$SEGMENTATION_TYPE" == *"diarized"* ]]; then
    VAD_COLUMN="${MVAD_DIR}.diarized.vad.npy"
  fi

  echo "Using VAD_COLUMN=$VAD_COLUMN"
  echo "wsds init $OUTPUT_BASE/$SEGMENTATION_TYPE --source_dataset=$OUTPUT_BASE/source --vad_column=$VAD_COLUMN"
  wsds init "$OUTPUT_BASE/$SEGMENTATION_TYPE" \
    --source_dataset="$OUTPUT_BASE/source" \
    --vad_column="$VAD_COLUMN"
fi

### summary ###
echo -e "\n${GRN}${BOLD} Conversion complete. Outputs are under: $OUTPUT_BASE${RST}\n"
echo -e "${BLU}${BOLD}=== Summary ===${RST}"
for key in "${!counts[@]}"; do
  echo -e "${YLW}$key:${RST} ${counts[$key]} wsds shards"
  total=$((total + counts[$key]))
done
echo -e "${GRN}${BOLD}Total:${RST} $total wsds shards\n"
