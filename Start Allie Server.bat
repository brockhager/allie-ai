@echo off
cd backend
call "%~dp0venv\Scripts\activate.bat"
uvicorn server:app --reload --host 0.0.0.0 --port 8001
pause