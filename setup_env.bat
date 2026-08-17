@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  R6 Siege Video Analyzer - Environment Setup
echo  GPU: RTX 3080 / CUDA 12.x
echo ============================================================

:: ── 1. Create venv ──────────────────────────────────────────────────────────
if not exist "venv" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/5] Virtual environment already exists, skipping.
)

call venv\Scripts\activate.bat

:: ── 2. Upgrade pip ──────────────────────────────────────────────────────────
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: ── 3. Install PyTorch with CUDA 12.1 ────────────────────────────────────────
echo [3/5] Installing PyTorch 2.x with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if !errorlevel! neq 0 (
    echo ERROR: PyTorch installation failed. Check your internet connection.
    pause
    exit /b 1
)

:: ── 4. Install project requirements ─────────────────────────────────────────
echo [4/5] Installing project requirements...
pip install -r requirements.txt --quiet
if !errorlevel! neq 0 (
    echo WARNING: Some requirements may have failed. Check output above.
)

:: bitsandbytes on Windows needs a special wheel
pip install bitsandbytes --prefer-binary --quiet 2>nul
if !errorlevel! neq 0 (
    echo WARNING: bitsandbytes not installed (quantization unavailable).
    echo          You can still run without --quantize flag.
)

:: ── 5. Verify installation ────────────────────────────────────────────────────
echo [5/5] Verifying CUDA / torch...
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT FOUND\"}')"
if !errorlevel! neq 0 (
    echo ERROR: PyTorch verification failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo  Activate with:  venv\Scripts\activate.bat
echo  Run test:       python test_pipeline.py
echo  Run analysis:   python analyze.py --video your_video.mp4
echo ============================================================
pause
