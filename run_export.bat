@echo off
cd /d "%~dp0"
echo Launching NanoDes Model Export...

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    python export_gui.py
) else (
    echo Error: Virtual environment 'venv' not found!
    pause
    exit /b 1
)

if %ERRORLEVEL% neq 0 (
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
