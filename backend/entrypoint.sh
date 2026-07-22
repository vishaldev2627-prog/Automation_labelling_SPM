#!/usr/bin/env bash
# Downloads the SAM2.1 checkpoint into the persistent volume on first boot
# (skipped on every later restart/redeploy once it's already there), then
# hands off to the real container command.
set -euo pipefail

CHECKPOINT_PATH="${SAM_CHECKPOINT:-/data/models/sam2.1_hiera_large.pt}"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

mkdir -p "$(dirname "$CHECKPOINT_PATH")"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "[entrypoint] SAM2 checkpoint not found at $CHECKPOINT_PATH — downloading (~900MB)..."
  curl -fL --retry 3 -o "${CHECKPOINT_PATH}.part" "$CHECKPOINT_URL"
  mv "${CHECKPOINT_PATH}.part" "$CHECKPOINT_PATH"
  echo "[entrypoint] Checkpoint downloaded."
else
  echo "[entrypoint] Checkpoint already present at $CHECKPOINT_PATH, skipping download."
fi

exec "$@"
