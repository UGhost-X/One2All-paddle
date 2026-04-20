#!/usr/bin/env python3
"""
嵌入式Python环境初始化脚本
使用uv管理依赖，从本地wheel安装torch和torchvision
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
EMBED_DIR = Path(__file__).parent.resolve()
WHEELS_DIR = PROJECT_ROOT / "wheels"

def setup_wheels_dir():
    """创建wheels目录并移动wheel文件"""
    WHEELS_DIR.mkdir(exist_ok=True)
    
    wheel_files = [
        PROJECT_ROOT / "torch-2.6.0+cu124-cp310-cp310-win_amd64.whl",
        PROJECT_ROOT / "torchvision-0.21.0+cu124-cp310-cp310-win_amd64.whl",
    ]
    
    for wheel in wheel_files:
        if wheel.exists():
            dest = WHEELS_DIR / wheel.name
            if not dest.exists():
                print(f"Moving {wheel.name} to wheels/")
                shutil.move(str(wheel), str(dest))
            else:
                print(f"{wheel.name} already in wheels/")

def find_uv():
    """查找uv可执行文件"""
    uv_exe = shutil.which("uv")
    if uv_exe:
        return uv_exe
    
    possible_paths = [
        os.path.expanduser("~/.cargo/bin/uv.exe"),
        os.path.expanduser("~/.cargo/bin/uv"),
        os.path.expandvars(r"C:\Users\%USERNAME%\.cargo\bin\uv.exe"),
        r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\Scripts\uv.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def init_uv_project():
    """初始化uv项目"""
    uv_exe = find_uv()
    if not uv_exe:
        print("Error: uv not found. Please install uv first.")
        print("Install: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        sys.exit(1)
    
    print(f"Found uv: {uv_exe}")
    
    os.chdir(EMBED_DIR)
    
    if not (EMBED_DIR / ".venv").exists():
        print("Creating virtual environment...")
        subprocess.run([uv_exe, "venv", "--python", "3.10"], check=True)
    
    print("Installing dependencies from Tsinghua mirror...")
    subprocess.run([uv_exe, "pip", "install", "-e", "."], check=True)
    
    print("Installing torch from local wheel...")
    torch_wheel = WHEELS_DIR / "torch-2.6.0+cu124-cp310-cp310-win_amd64.whl"
    if torch_wheel.exists():
        subprocess.run([
            uv_exe, "pip", "install", 
            "--force-reinstall",
            str(torch_wheel)
        ], check=True)
    else:
        print(f"Warning: {torch_wheel} not found")
    
    print("Installing torchvision from local wheel...")
    torchvision_wheel = WHEELS_DIR / "torchvision-0.21.0+cu124-cp310-cp310-win_amd64.whl"
    if torchvision_wheel.exists():
        subprocess.run([
            uv_exe, "pip", "install",
            "--force-reinstall",
            str(torchvision_wheel)
        ], check=True)
    else:
        print(f"Warning: {torchvision_wheel} not found")
    
    print("Environment setup complete!")

def create_env_file():
    """创建.env配置文件"""
    env_file = PROJECT_ROOT / ".env"
    
    if env_file.exists():
        print(f".env file already exists at {env_file}")
        return
    
    env_content = """# One2All Paddle 配置文件
# 模型输出目录（默认：项目根目录/output）
OUTPUT_DIR=output

# 产品数据目录（默认：项目根目录/product）
PRODUCT_DIR=product

# 预训练模型目录（默认：项目根目录/models/pretrained）
MODELS_DIR=models/pretrained

# 推理服务脚本目录（默认：项目根目录/inference_services）
INFERENCE_SCRIPTS_DIR=inference_services
"""
    
    env_file.write_text(env_content, encoding="utf-8")
    print(f"Created .env file at {env_file}")

def create_launcher():
    """创建启动脚本"""
    # 创建数据目录
    output_dir = PROJECT_ROOT / "output"
    product_dir = PROJECT_ROOT / "product"
    output_dir.mkdir(exist_ok=True)
    product_dir.mkdir(exist_ok=True)
    
    start_bat = EMBED_DIR / "start.bat"
    start_bat.write_text("""@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\\Scripts\\activate.bat
set PYTHONPATH=%~dp0\\..;%PYTHONPATH%
python "%~dp0\\..\\main.py"
pause
""")
    
    start_service_bat = EMBED_DIR / "start_service.bat"
    start_service_bat.write_text("""@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\\Scripts\\activate.bat >nul 2>&1
set PYTHONPATH=%~dp0\\..;%PYTHONPATH%
start /b "" python "%~dp0\\..\\main.py" >nul 2>&1
echo Service started on http://localhost:8000
echo.
echo Configuration file: .env
echo Edit .env to customize output/product directories
""")
    
    print("Launcher scripts created.")

def main():
    print("=" * 60)
    print("One2All Paddle Embedded Environment Setup")
    print("=" * 60)
    
    setup_wheels_dir()
    init_uv_project()
    create_env_file()
    create_launcher()
    
    print("\n" + "=" * 60)
    print("Setup complete! Run 'start.bat' to start the service.")
    print("=" * 60)
    print("\nTo customize paths, edit the .env file in project root.")

if __name__ == "__main__":
    main()
