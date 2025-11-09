@echo off
echo ========================================
echo Quick Teach - Help Allie Learn Faster!
echo ========================================
echo.

call "%~dp0..\venv\Scripts\activate.bat"
python "%~dp0quick_teach.py" %*

pause
