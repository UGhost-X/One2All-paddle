"""
项目配置模块
支持通过.env文件配置路径
"""
import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env():
    """加载.env文件，优先级：项目根目录 > 当前工作目录"""
    project_root = Path(__file__).parent.parent.resolve()
    
    # 尝试从项目根目录加载
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        return
    
    # 尝试从当前工作目录加载
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file, override=True)


# 启动时加载.env
_load_env()


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.resolve()


def get_output_dir() -> Path:
    """
    获取输出目录路径
    优先级：.env文件 OUTPUT_DIR > 默认值 output/
    """
    output_dir = os.environ.get("OUTPUT_DIR", "output")
    path = Path(output_dir)
    if not path.is_absolute():
        path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_product_dir() -> Path:
    """
    获取产品数据目录路径
    优先级：.env文件 PRODUCT_DIR > 默认值 product/
    """
    product_dir = os.environ.get("PRODUCT_DIR", "product")
    path = Path(product_dir)
    if not path.is_absolute():
        path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    """
    获取预训练模型目录路径
    优先级：.env文件 MODELS_DIR > 默认值 models/pretrained/
    """
    models_dir = os.environ.get("MODELS_DIR", "models/pretrained")
    path = Path(models_dir)
    if not path.is_absolute():
        path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_inference_scripts_dir() -> Path:
    """
    获取推理服务脚本目录路径
    优先级：.env文件 INFERENCE_SCRIPTS_DIR > 默认值 inference_services/
    """
    scripts_dir = os.environ.get("INFERENCE_SCRIPTS_DIR", "inference_services")
    path = Path(scripts_dir)
    if not path.is_absolute():
        path = get_project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


class PathConfig:
    """路径配置类，提供统一的路径访问接口"""
    
    @property
    def output_dir(self) -> Path:
        return get_output_dir()
    
    @property
    def product_dir(self) -> Path:
        return get_product_dir()
    
    @property
    def models_dir(self) -> Path:
        return get_models_dir()
    
    @property
    def inference_scripts_dir(self) -> Path:
        return get_inference_scripts_dir()
    
    def get_project_output_path(self, project_id: str, task_uuid: str = None) -> Path:
        """获取项目输出路径"""
        path = self.output_dir / str(project_id)
        if task_uuid:
            path = path / str(task_uuid)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_project_product_path(self, project_id: str, task_uuid: str = None) -> Path:
        """获取项目产品数据路径"""
        path = self.product_dir / str(project_id) / "train"
        if task_uuid:
            path = path / str(task_uuid)
        path.mkdir(parents=True, exist_ok=True)
        return path


path_config = PathConfig()
