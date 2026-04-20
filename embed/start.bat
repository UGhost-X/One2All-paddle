@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=%~dp0\..;%PYTHONPATH%
.venv-embed\Scripts\python.exe "%~dp0\..\main.py"
pause
