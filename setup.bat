@echo off
setlocal enabledelayedexpansion

echo == Railway Segmentation Annotation Tool: setup ==

where python >nul 2>nul
if errorlevel 1 (
    echo python is required and was not found on PATH.
    exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
    echo Node.js ^(^>=18^) is required and was not found on PATH.
    exit /b 1
)

set ROOT_DIR=%~dp0

echo -- Backend: creating virtualenv --
cd /d "%ROOT_DIR%backend"
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist "%ROOT_DIR%sam2_repo" (
    echo -- Cloning SAM2 --
    git clone --depth 1 https://github.com/facebookresearch/sam2.git "%ROOT_DIR%sam2_repo"
    pip install -e "%ROOT_DIR%sam2_repo"
) else (
    echo -- SAM2 repo already present, skipping clone --
)

if not exist "%ROOT_DIR%models" mkdir "%ROOT_DIR%models"
if not exist "%ROOT_DIR%models\sam2.1_hiera_large.pt" (
    echo -- Downloading SAM2.1 checkpoint ^(~900MB^) --
    curl -L -o "%ROOT_DIR%models\sam2.1_hiera_large.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
) else (
    echo -- Checkpoint already present, skipping download --
)

if not exist ".env" (
    copy .env.example .env
    echo -- Created backend\.env from template. Edit DATASET_PATH before running. --
)

call .venv\Scripts\deactivate.bat

echo -- Frontend: installing dependencies --
cd /d "%ROOT_DIR%frontend"
call npm install

if not exist "%ROOT_DIR%exports" mkdir "%ROOT_DIR%exports"
if not exist "%ROOT_DIR%logs" mkdir "%ROOT_DIR%logs"

echo.
echo Setup complete.
echo 1. Edit backend\.env and set DATASET_PATH to your dataset folder.
echo 2. Start the backend:  cd backend ^&^& .venv\Scripts\activate ^&^& python run.py
echo 3. Start the frontend: cd frontend ^&^& npm run dev
echo 4. Open http://localhost:5173

endlocal
