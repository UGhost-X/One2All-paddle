# One2All Paddle 嵌入式Python环境

## 环境说明

本目录包含嵌入式Python环境的配置，使用 `uv` 进行依赖管理。

## 快速开始

### 1. 确保已安装uv

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 运行环境初始化

```powershell
cd embed
python setup_env.py
```

此脚本会：
- 创建虚拟环境
- 从清华镜像安装其他依赖
- 从本地wheel安装torch和torchvision
- 创建启动脚本

### 3. 启动服务

```powershell
# 交互式启动（带控制台窗口）
.\start.bat

# 后台启动
.\start_service.bat
```

## 目录结构

```
embed/
├── pyproject.toml      # 依赖配置（清华镜像源）
├── setup_env.py        # 环境初始化脚本
├── start.bat           # 交互式启动脚本
├── start_service.bat   # 后台服务启动脚本
├── .venv/              # 虚拟环境（自动生成）
└── README.md           # 本文件
```

## 依赖安装说明

- **其他依赖**: 从清华镜像源安装
- **torch**: 从 `../wheels/torch-2.6.0+cu124-cp310-cp310-win_amd64.whl` 安装
- **torchvision**: 从 `../wheels/torchvision-0.21.0+cu124-cp310-cp310-win_amd64.whl` 安装

## 手动安装依赖

如果需要手动安装：

```powershell
cd embed
uv venv --python 3.10
uv pip install -e .
uv pip install --force-reinstall ../wheels/torch-2.6.0+cu124-cp310-cp310-win_amd64.whl
uv pip install --force-reinstall ../wheels/torchvision-0.21.0+cu124-cp310-cp310-win_amd64.whl
```
