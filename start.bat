@echo off
setlocal

echo == Railway Segmentation Annotation Tool: start ==

set ROOT_DIR=%~dp0

if not exist "%ROOT_DIR%backend\.venv\Scripts\activate.bat" (
    echo Backend virtualenv not found. Run setup.bat first.
    exit /b 1
)
if not exist "%ROOT_DIR%frontend\node_modules" (
    echo Frontend dependencies not found. Run setup.bat first.
    exit /b 1
)

echo -- Starting backend in a new window (http://localhost:8000) --
start "Railway Annotator - Backend" cmd /k "cd /d "%ROOT_DIR%backend" && call .venv\Scripts\activate.bat && python run.py"

echo -- Starting frontend in a new window (http://localhost:5173) --
start "Railway Annotator - Frontend" cmd /k "cd /d "%ROOT_DIR%frontend" && npm run dev"

echo.
echo Two windows were opened - keep them running while you work:
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo Close those windows (or Ctrl+C inside them) to stop the servers.
echo.
echo Opening the app in your browser...
timeout /t 3 /nobreak >nul
start http://localhost:5173

endlocal
