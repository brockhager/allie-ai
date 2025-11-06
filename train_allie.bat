@echo off
REM Activate the virtual environment
call "%~dp0venv\Scripts\activate.bat"

REM Run the training script
python train_allie.py

REM Keep the window open so you can see output
pause
