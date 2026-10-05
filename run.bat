@echo off
cd /d "%~dp0"
echo Launching NanoDes...

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    python gui.py
) else (
    echo Error: Virtual environment 'venv' not found!
    pause
    exit /b 1
)

if %ERRORLEVEL% neq 0 (
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
