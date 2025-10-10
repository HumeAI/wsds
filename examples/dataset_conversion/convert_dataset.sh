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

YT_DATA_SPECIFIC=$(to_bool "$(yq -r '.flags.yt_data_specific' "$CFG")")
REQUIRES_SORTING=$(to_bool "$(yq -r '.flags.audio_requires_sorting' "$CFG")")
MIXED_AUDIO=$(to_bool "$(yq -r '.flags.mixed_audio' "$CFG")")


MVAD_DIR_NAME=$(yq -r '.flags.mvad_dir_name // "mvad"' "$CFG")

AUDIO_BASE=$(yq -r '.paths.audio_base' "$CFG")
INPUT_BASE=$(yq -r '.paths.input_base' "$CFG")
OUTPUT_BASE=$(yq -r '.paths.output_base' "$CFG")

SEGMENTATION_TYPE=$(yq -r '.segmentation_type' "$CFG")

mapfile -t SUBDIRS < <(yq -r '.subdirs[]' "$CFG")

MVAD_DIR="${SEGMENTATION_TYPE}_mvad"

RUN_CLEANUP=$(to_bool "$(yq -r '.flags.run_cleanup // "false"' "$CFG")")
CLEANUP_DRY_RUN=$(to_bool "$(yq -r '.flags.cleanup_dry_run // "true"' "$CFG")")

mkdir -p "$OUTPUT_BASE/source/audio" "$OUTPUT_BASE/source/$MVAD_DIR" "$OUTPUT_BASE/$SEGMENTATION_TYPE"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; BOLD='\033[1m'; RST='\033[0m'

declare -A counts
total=0

# build common args
COMMON_ARGS=()
if [[ "$YT_DATA_SPECIFIC" == "true" ]]; then
  COMMON_ARGS+=(--yt_data_specific)
fi

if [[ "$REQUIRES_SORTING" == "true" ]]; then
  COMMON_ARGS+=(--audio_requires_sorting)
fi

if [[ "$MIXED_AUDIO" == "true" ]]; then
  COMMON_ARGS+=(--mixed_audio)
fi




### audio ###
if [[ "$CONVERT_AUDIO" == "true" ]]; then
  echo -e "${BLU}${BOLD}=== Converting audio ===${RST}"

  mapfile -t AUDIO_TARS < <(
    for f in "$AUDIO_BASE"/*.tar; do
      base=$(basename "${f}")
      out="$OUTPUT_BASE/source/audio/${base%.tar*}.wsds"
      [[ -f "$out" ]] || echo "$f"
    done
  )

  if (( ${#AUDIO_TARS[@]} > 0 )); then
    parallel --plus --tag --bar \
      wsds shard_from_webdataset {} "$OUTPUT_BASE/source/audio/{/.}.wsds" \
      "${COMMON_ARGS[@]}" \
      ::: "${AUDIO_TARS[@]}"
  else
    echo "No new audio tars to process."
  fi

  counts["source/audio"]=$(ls -1 "$OUTPUT_BASE/source/audio"/*.wsds 2>/dev/null | wc -l || echo 0)

  # Rename .source_separation.wsds → .wsds
  for f in "$OUTPUT_BASE/source/audio"/*.wsds; do
    if [[ "$f" == *source_separation* ]]; then
      new_name="${f/source_separation./}"
      if [[ "$f" != "$new_name" ]]; then
        mv "$f" "$new_name"
      fi
    fi
  done
fi

### mvad ###
if [[ "$CONVERT_MVAD" == "true" ]]; then
  echo -e "${BLU}${BOLD}=== Converting $MVAD_DIR ===${RST}"

  mapfile -t MVAD_TARS < <(
    for f in "$INPUT_BASE/$MVAD_DIR_NAME"/*.tar.gz; do
        base=$(basename "${f}")
        # remove trailing .tar.gz or .tar
        base_noext="${base%.tar.gz}"
        base_noext="${base_noext%.tar}"
        base_noext="${base_noext%.mvad}"
        out="$OUTPUT_BASE/source/$MVAD_DIR/${base_noext}.wsds"
        [[ -f "$out" ]] || echo "$f"
    done
  )

  if (( ${#MVAD_TARS[@]} > 0 )); then
    parallel --plus --tag --bar \
      wsds shard_from_webdataset {} "$OUTPUT_BASE/source/$MVAD_DIR/{/...}.wsds" \
      ::: "${MVAD_TARS[@]}"
  else
    echo "No new mvad tars to process."
  fi

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

    mapfile -t ART_TARS < <(
      for f in "$in_dir"/*.tar.gz "$in_dir"/*.tar; do
        # Skip if the glob didn't expand (file doesn't exist)
        [[ -f "$f" ]] || continue
        base=$(basename "${f}")
        base_root="${base%%.*}"   # keep only before the first dot
        out="$out_dir/${base_root}.wsds"
        [[ -f "$out" ]] || echo "$f"
      done
    )

    if (( ${#ART_TARS[@]} > 0 )); then
      parallel --plus --tag --bar \
        wsds shard_from_webdataset {} "$out_dir/{/...}.wsds" \
        --compression no-compression \
        ::: "${ART_TARS[@]}"
    else
      echo "No new artifact tars to process for $sub."
    fi

    counts["$SEGMENTATION_TYPE/$sub"]=$(ls -1 "$out_dir"/*.wsds 2>/dev/null | wc -l || echo 0)
  done

  # Rename source_separation to isolated_audio if it exists
  source_sep_dir="$OUTPUT_BASE/$SEGMENTATION_TYPE/source_separation"
  isolated_audio_dir="$OUTPUT_BASE/$SEGMENTATION_TYPE/isolated_audio"
  if [[ -d "$source_sep_dir" ]]; then
    echo -e "${BLU}${BOLD}=== Renaming source_separation to isolated_audio ===${RST}"
    mv "$source_sep_dir" "$isolated_audio_dir"
    echo "Renamed: $source_sep_dir -> $isolated_audio_dir"
  fi
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

### cleanup ###
if [[ "$RUN_CLEANUP" == "true" ]]; then
  echo -e "${BLU}${BOLD}=== Running sync/clean ===${RST}"

  python3 - "$OUTPUT_BASE" "$SEGMENTATION_TYPE" "$CLEANUP_DRY_RUN" "${SUBDIRS[@]}" <<'EOF'
import os, sys

def sync_and_clean(base_dir, other_dirs, dry_run=True):
    base_files = set(os.listdir(base_dir))
    print(f"[INFO] Found {len(base_files)} files in base dir: {base_dir}")

    for other in other_dirs:
        if not os.path.exists(other):
            print(f"[WARN] Skipping missing dir: {other}")
            continue

        other_files = set(os.listdir(other))
        print(f"[INFO] Checking {other} ({len(other_files)} files)")

        extra_files = other_files - base_files
        if not extra_files:
            print(f"  ✅ No extra files in {other}")
            continue

        for f in sorted(extra_files):
            file_path = os.path.join(other, f)
            if dry_run:
                print(f"  [DRY RUN] Would remove: {file_path}")
            else:
                print(f"  🗑 Removing: {file_path}")
                os.remove(file_path)

# args from bash
output_base = sys.argv[1]
segmentation_type = sys.argv[2]
dry_run = sys.argv[3].lower() == "true"
subdirs = sys.argv[4:]

base_dir = f"{output_base}/source/audio"
artifact_dirs = [f"{output_base}/{segmentation_type}/{sub}" for sub in subdirs]
artifact_dirs.insert(0, f"{output_base}/source/{segmentation_type}_mvad")  # include mvad

sync_and_clean(base_dir, artifact_dirs, dry_run=dry_run)
EOF
fi

