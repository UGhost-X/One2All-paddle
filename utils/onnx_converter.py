"""
Paddle 模型到 ONNX 格式转换工具模块
提供将 PaddlePaddle 训练模型转换为 ONNX 格式的接口
"""
import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

ONNX_MODEL_FILENAME = "model.onnx"
CONFIG_FILENAME = "deploy.yaml"
MODEL_FILE_PREFIX = "model"


def check_paddle2onnx_available() -> bool:
    """检查 paddle2onnx 是否可用"""
    try:
        import paddle2onnx
        return True
    except ImportError:
        return False


def get_paddle_model_paths(model_dir: str) -> Dict[str, Any]:
    """获取 Paddle 模型文件路径"""
    model_dir = Path(model_dir)
    model_paths = {}
    
    # 首先检查 inference 目录（PaddleX 导出格式）
    inference_dir = model_dir / "inference"
    if inference_dir.exists():
        if (inference_dir / "inference.json").exists():
            model_paths["model_file"] = str(inference_dir / "inference.json")
        elif (inference_dir / "inference.pdmodel").exists():
            model_paths["model_file"] = str(inference_dir / "inference.pdmodel")
        
        if (inference_dir / "inference.pdiparams").exists():
            model_paths["params_file"] = str(inference_dir / "inference.pdiparams")
        
        if model_paths:
            model_paths["model_dir"] = str(inference_dir)
            return model_paths
    
    # 检查标准模型文件格式
    if (model_dir / f"{MODEL_FILE_PREFIX}.json").exists():
        model_paths["model_file"] = str(model_dir / f"{MODEL_FILE_PREFIX}.json")
    elif (model_dir / f"{MODEL_FILE_PREFIX}.pdmodel").exists():
        model_paths["model_file"] = str(model_dir / f"{MODEL_FILE_PREFIX}.pdmodel")
    
    if (model_dir / f"{MODEL_FILE_PREFIX}.pdiparams").exists():
        model_paths["params_file"] = str(model_dir / f"{MODEL_FILE_PREFIX}.pdiparams")
    
    return model_paths


def convert_paddle_to_onnx(
    paddle_model_dir: str,
    output_dir: Optional[str] = None,
    opset_version: int = 11,
    simplify: bool = True,
    quantize: bool = False,
    input_shape: Optional[Dict[str, tuple]] = None,
) -> Dict[str, Any]:
    """
    将 PaddlePaddle 模型转换为 ONNX 格式
    
    Args:
        paddle_model_dir: Paddle 模型目录路径
        output_dir: ONNX 模型输出目录 (默认与输入目录相同)
        opset_version: ONNX opset 版本 (默认 11)
        simplify: 是否简化 ONNX 模型
        quantize: 是否量化模型
        input_shape: 输入形状字典，格式: {"image": [1, 3, 640, 640]}
    
    Returns:
        包含转换结果的字典
    """
    paddle_model_dir = Path(paddle_model_dir)
    
    if output_dir is None:
        output_dir = paddle_model_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    model_paths = get_paddle_model_paths(str(paddle_model_dir))
    
    if not model_paths:
        return {
            "success": False,
            "error": "未找到 Paddle 模型文件，请检查模型目录"
        }
    
    if "model_file" not in model_paths or "params_file" not in model_paths:
        return {
            "success": False,
            "error": "模型文件不完整，需要 model.pdmodel 和 model.pdiparams"
        }
    
    # 获取实际的模型目录（可能是 inference 子目录）
    actual_model_dir = model_paths.get("model_dir", str(paddle_model_dir))
    onnx_model_path = output_dir / ONNX_MODEL_FILENAME
    
    try:
        # 使用 paddle2onnx 命令行工具进行转换
        import subprocess
        
        cmd = [
            "paddle2onnx",
            "--model_dir", str(Path(model_paths["model_file"]).parent),
            "--model_filename", Path(model_paths["model_file"]).name,
            "--params_filename", Path(model_paths["params_file"]).name,
            "--save_file", str(onnx_model_path),
            "--opset_version", str(opset_version),
            "--enable_onnx_checker", "False",
        ]
        
        if input_shape:
            input_shape_list = list(input_shape.values())[0] if input_shape else None
            if input_shape_list:
                cmd.extend(["--input_shape_dict", str(input_shape_list)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if simplify:
            try:
                import onnx
                from onnx import shape_inference
                
                onnx_model = onnx.load(str(onnx_model_path))
                onnx_model = shape_inference.infer_shapes(onnx_model)
                
                from onnxsim import simplify as onnx_simplify
                onnx_model, check = onnx_simplify(onnx_model)
                onnx.save(onnx_model, str(onnx_model_path))
                logger.info("ONNX 模型简化完成")
            except ImportError:
                logger.warning("onnxsim 未安装，跳过模型简化")
            except Exception as e:
                logger.warning(f"模型简化失败: {e}")
        
        if not onnx_model_path.exists():
            return {
                "success": False,
                "error": "ONNX 模型文件未生成"
            }
        
        # 尝试从实际模型目录或父目录复制配置文件
        config_src = Path(actual_model_dir) / CONFIG_FILENAME
        if not config_src.exists():
            config_src = paddle_model_dir / CONFIG_FILENAME
        if config_src.exists():
            shutil.copy(config_src, output_dir / CONFIG_FILENAME)
        
        return {
            "success": True,
            "onnx_model_path": str(onnx_model_path),
            "output_dir": str(output_dir),
            "opset_version": opset_version,
            "model_size": os.path.getsize(onnx_model_path)
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"ONNX 转换失败: {e.stderr}")
        return {
            "success": False,
            "error": f"ONNX 转换失败: {e.stderr}"
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "paddle2onnx 命令未找到，请安装: pip install paddle2onnx"
        }
    except Exception as e:
        logger.error(f"ONNX 转换失败: {e}")
        return {
            "success": False,
            "error": f"ONNX 转换失败: {str(e)}"
        }


def convert_with_cli(
    paddle_model_dir: str,
    output_dir: Optional[str] = None,
    opset_version: int = 11,
) -> Dict[str, Any]:
    """
    使用 paddle2onnx 命令行工具进行转换
    
    Args:
        paddle_model_dir: Paddle 模型目录路径
        output_dir: ONNX 模型输出目录
        opset_version: ONNX opset 版本
    
    Returns:
        转换结果字典
    """
    paddle_model_dir = Path(paddle_model_dir)
    
    if output_dir is None:
        output_dir = paddle_model_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    model_paths = get_paddle_model_paths(str(paddle_model_dir))
    
    if not model_paths or "model_file" not in model_paths or "params_file" not in model_paths:
        return {
            "success": False,
            "error": "模型文件不完整"
        }
    
    onnx_model_path = output_dir / ONNX_MODEL_FILENAME
    
    try:
        cmd = [
            "paddle2onnx",
            "--model_dir", str(paddle_model_dir),
            "--model_filename", Path(model_paths["model_file"]).name,
            "--params_filename", Path(model_paths["params_file"]).name,
            "--save_file", str(onnx_model_path),
            "--opset_version", str(opset_version),
            "--enable_validation", "False",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if onnx_model_path.exists():
            return {
                "success": True,
                "onnx_model_path": str(onnx_model_path),
                "output_dir": str(output_dir),
                "opset_version": opset_version
            }
        else:
            return {
                "success": False,
                "error": "模型文件未生成"
            }
            
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"转换失败: {e.stderr}"
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "paddle2onnx 命令未找到，请安装: pip install paddle2onnx"
        }


class ONNXConverter:
    """ONNX 转换器类"""
    
    def __init__(self, paddle_model_dir: str):
        self.paddle_model_dir = Path(paddle_model_dir)
        self.model_paths = get_paddle_model_paths(str(self.paddle_model_dir))
    
    def convert(
        self,
        output_dir: Optional[str] = None,
        opset_version: int = 11,
        simplify: bool = True,
        input_shape: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, Any]:
        """
        执行模型转换
        
        Args:
            output_dir: 输出目录
            opset_version: ONNX opset 版本
            simplify: 是否简化模型
            input_shape: 输入形状
        
        Returns:
            转换结果
        """
        if check_paddle2onnx_available():
            return convert_paddle_to_onnx(
                str(self.paddle_model_dir),
                output_dir,
                opset_version,
                simplify,
                input_shape=input_shape
            )
        else:
            return convert_with_cli(
                str(self.paddle_model_dir),
                output_dir,
                opset_version
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model_paths:
            return {"valid": False, "error": "未找到有效的 Paddle 模型"}
        
        return {
            "valid": True,
            "model_file": self.model_paths.get("model_file"),
            "params_file": self.model_paths.get("params_file"),
            "model_dir": str(self.paddle_model_dir)
        }
