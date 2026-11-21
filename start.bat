@echo off
setlocal

rem ================= CONFIG =================
rem Python interpreter (leave as "python" to use PATH)
set "PYTHON=python"
rem Server host and port
set "HOST=127.0.0.1"
set "PORT=8000"
rem Auto-reload on code changes: 1 = on, 0 = off
set "RELOAD=1"
rem Open browser automatically: 1 = on, 0 = off
set "OPEN_BROWSER=1"
rem ==========================================

cd /d "%~dp0"

where %PYTHON% >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found: %PYTHON%
    echo Install Python or edit the PYTHON variable at the top of this script.
    pause
    exit /b 1
)

%PYTHON% -c "import fastapi, uvicorn, torch, PIL, numpy" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Missing dependencies. Run:
    echo     %PYTHON% -m pip install fastapi uvicorn torch pillow numpy
    pause
    exit /b 1
)

if not exist "dl_model\checkpoints\mnist_classifier.pt" (
    echo [ERROR] Model checkpoint not found: dl_model\checkpoints\mnist_classifier.pt
    pause
    exit /b 1
)

set "UVICORN_ARGS=app.main:app --host %HOST% --port %PORT%"
if "%RELOAD%"=="1" set "UVICORN_ARGS=%UVICORN_ARGS% --reload"

echo ================================================
echo  Handwritten Digit Recognition
echo  URL: http://%HOST%:%PORT%
echo  Press Ctrl+C to stop the server
echo ================================================
echo.

if "%OPEN_BROWSER%"=="1" (
    start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start http://%HOST%:%PORT%"
)

%PYTHON% -m uvicorn %UVICORN_ARGS%

echo.
echo Server stopped.
pause
