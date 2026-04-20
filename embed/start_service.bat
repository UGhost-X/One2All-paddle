@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=%~dp0\..;%PYTHONPATH%
start /b "" .venv-embed\Scripts\python.exe "%~dp0\..\main.py" >nul 2>&1
echo Service started on http://localhost:8000
echo.
echo Configuration file: .env
echo Edit .env to customize output/product directories
