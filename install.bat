@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================
echo  Fooocus Windows setup
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found on PATH.
    echo.
    echo 1. Install Python 3.10 or 3.11 from:
    echo    https://www.python.org/downloads/windows/
    echo 2. During setup, check "Add python.exe to PATH"
    echo 3. Open a new Command Prompt and run this file again.
    echo.
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
    echo Unsupported Python version. Please use Python 3.10, 3.11, or 3.12.
    python --version
    pause
    exit /b 1
)

if not exist "fooocus_env\Scripts\python.exe" (
    echo Creating virtual environment: fooocus_env
    python -m venv fooocus_env
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Using existing virtual environment: fooocus_env
)

echo.
echo Upgrading pip...
"fooocus_env\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools

echo.
echo Installing PyTorch with CUDA 12.1 support...
"fooocus_env\Scripts\python.exe" -m pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo CUDA PyTorch install failed. Installing CPU-only PyTorch instead...
    "fooocus_env\Scripts\python.exe" -m pip install torch==2.1.0 torchvision==0.16.0
    if errorlevel 1 (
        echo PyTorch install failed.
        pause
        exit /b 1
    )
)

echo.
echo Installing Fooocus requirements (Gradio 4.44.1)...
"fooocus_env\Scripts\python.exe" -m pip install -r requirements_versions.txt
if errorlevel 1 (
    echo Requirements install failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup finished.
echo  Next: double-click run.bat
echo  First launch downloads models automatically.
echo ============================================
pause
endlocal
