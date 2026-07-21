#!/usr/bin/env bash
# Local (non-Docker) setup for the Railway Segmentation Annotation Tool.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== Railway Segmentation Annotation Tool: setup =="

command -v python3 >/dev/null 2>&1 || { echo "python3 is required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js (>=18) is required"; exit 1; }

echo "-- Backend: creating virtualenv --"
cd "$ROOT_DIR/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -d "../sam2_repo" ]; then
  echo "-- Cloning SAM2 --"
  git clone --depth 1 https://github.com/facebookresearch/sam2.git "$ROOT_DIR/sam2_repo"
  pip install -e "$ROOT_DIR/sam2_repo"
else
  echo "-- SAM2 repo already present, skipping clone --"
fi

mkdir -p "$ROOT_DIR/models"
if [ ! -f "$ROOT_DIR/models/sam2.1_hiera_large.pt" ]; then
  echo "-- Downloading SAM2.1 checkpoint (this is ~900MB) --"
  curl -L -o "$ROOT_DIR/models/sam2.1_hiera_large.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
else
  echo "-- Checkpoint already present, skipping download --"
fi

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "-- Created backend/.env from template. Edit DATASET_PATH before running. --"
fi

deactivate

echo "-- Frontend: installing dependencies --"
cd "$ROOT_DIR/frontend"
npm install

mkdir -p "$ROOT_DIR/exports" "$ROOT_DIR/logs"

echo ""
echo "Setup complete."
echo "1. Edit backend/.env and set DATASET_PATH to your dataset folder."
echo "2. Start the backend:  cd backend && source .venv/bin/activate && python run.py"
echo "3. Start the frontend: cd frontend && npm run dev"
echo "4. Open http://localhost:5173"
