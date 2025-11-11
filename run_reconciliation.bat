@echo off
REM Run reconciliation worker once
REM Change to the directory where this script lives
pushd "%~dp0"

REM Try common virtualenv locations
if exist ".venv\Scripts\activate.bat" (
	echo Activating .venv\Scripts\activate.bat
	call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
	echo Activating venv\Scripts\activate.bat
	call "venv\Scripts\activate.bat"
) else (
	echo No local virtualenv found at .venv or venv. Using system Python.
)

echo Running reconciliation worker...
python scripts\reconciliation_worker.py --once

echo Reconciliation worker completed.
popd