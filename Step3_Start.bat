@echo off
REM Check if Python is installed
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python is not installed or not added to PATH.
    pause
    exit /b
)

REM Check if the virtual environment exists
IF NOT EXIST "venv\Scripts\activate" (
    echo Virtual environment not found. Please create it first.
    pause
    exit /b
)

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Check if app.py exists
IF NOT EXIST "app.py" (
    echo app.py not found in the current directory.
    pause
    exit /b
)

REM Run app.py
echo Running app.py...
python app.py

REM Pause to keep the terminal open after the app exits
pause
