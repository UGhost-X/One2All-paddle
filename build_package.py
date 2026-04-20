#!/usr/bin/env python3
"""
打包脚本：创建可分发部署包（包含嵌入式Python）
"""
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"

def create_package():
    DIST_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)
    
    package_dir = BUILD_DIR / "One2All-Paddle"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # 复制项目文件（不包含embed目录，单独处理）
    # 注意：wheels/ 不打包，因为依赖已安装到虚拟环境中
    includes = [
        "main.py",
        "pyproject.toml",
        ".env",
        ".env.example",
        "routers/",
        "services/",
        "utils/",
        "templates/",
        "models/",
        "output/",
        "product/",
        "inference_services/",
        "README.md",
        # 嵌入式Python
        "python-3.10.0rc2-embed-amd64/",
    ]
    
    for item in includes:
        src = PROJECT_ROOT / item
        dst = package_dir / item
        if not src.exists():
            print(f"Warning: {item} not found, skipping")
            continue
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            # 复制目录，排除不必要的文件
            ignore_patterns = shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.egg-info",
                ".git", ".gitignore",
                "*.log", "*.tmp"
            )
            shutil.copytree(src, dst, ignore=ignore_patterns)
    
    # 单独处理 embed 目录（只复制必要文件，不包含.venv）
    embed_src = PROJECT_ROOT / "embed"
    embed_dst = package_dir / "embed"
    embed_dst.mkdir(parents=True, exist_ok=True)
    
    # 复制 embed 目录下的文件（非目录）
    for item in embed_src.iterdir():
        if item.is_file():
            shutil.copy2(item, embed_dst / item.name)
    
    # 复制 .venv-embed 虚拟环境
    venv_src = embed_src / ".venv-embed"
    venv_dst = embed_dst / ".venv-embed"
    if venv_src.exists():
        print("Copying virtual environment...")
        shutil.copytree(
            venv_src, venv_dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    
    # 创建 ZIP
    zip_path = DIST_DIR / "One2All-Paddle-Deploy.zip"
    if zip_path.exists():
        zip_path.unlink()
    
    print("Creating ZIP package...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zf.write(file_path, arcname)
    
    print(f"Package created: {zip_path}")
    print(f"Package size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nDeploy instructions:")
    print("1. Unzip One2All-Paddle-Deploy.zip")
    print("2. Edit .env file to configure paths")
    print("3. Run embed/start.bat or embed/start_service.bat")

def update_venv_in_build():
    """增量更新：只更新build目录中的虚拟环境"""
    package_dir = BUILD_DIR / "One2All-Paddle"
    if not package_dir.exists():
        print("Build directory not found, running full package...")
        create_package()
        return
    
    # 更新虚拟环境
    venv_src = PROJECT_ROOT / "embed" / ".venv-embed"
    venv_dst = package_dir / "embed" / ".venv-embed"
    
    if venv_src.exists():
        print("Updating virtual environment in build directory...")
        if venv_dst.exists():
            shutil.rmtree(venv_dst)
        shutil.copytree(
            venv_src, venv_dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        print("Virtual environment updated!")
    
    # 重新打包ZIP
    zip_path = DIST_DIR / "One2All-Paddle-Deploy.zip"
    if zip_path.exists():
        zip_path.unlink()
    
    print("Creating ZIP package...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zf.write(file_path, arcname)
    
    print(f"Package created: {zip_path}")
    print(f"Package size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    import sys
    if "--update" in sys.argv:
        update_venv_in_build()
    else:
        create_package()
