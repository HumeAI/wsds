#!/usr/bin/env bash
# Run the full test suite in isolated environments, one per decoder backend.
#
# Usage:
#   ./scripts/test_audio_backends.sh
#
# Each environment installs only one decoder backend so we verify that the
# interfaces work correctly regardless of which backend is present.

TEST_COMMAND=${1:-"python -m tests"}

parallel --tag --lb \
    "uv run --isolated --with {} $TEST_COMMAND" \
    ::: humecodec "torchaudio<2.9" torchcodec
