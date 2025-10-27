@echo off
echo JARVIS Desktop Application Launcher
echo ===================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first: python setup_venv.py
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if PyWebView is installed
python -c "import webview" 2>nul
if errorlevel 1 (
    echo Installing PyWebView...
    pip install pywebview
)

REM Launch JARVIS Desktop Application
echo Launching JARVIS Desktop Application...
python jarvis_desktop.py

pause
