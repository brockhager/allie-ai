@echo off
REM Start Allie server (robust for different venv layouts)
REM Usage: double-click or run from a shell. This will try to activate .venv then venv, then fall back to system Python.

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

echo Starting Allie server using start_server.py (preferred)...
python "%~dp0start_server.py"

if %ERRORLEVEL% neq 0 (
	echo.
	echo start_server.py failed with exit code %ERRORLEVEL%. Trying direct uvicorn in backend/
	pushd backend
	python -m uvicorn server:app --reload --host 0.0.0.0 --port 8001
	popd
)

echo.
echo Press any key to close this window...
pause >nul
popd