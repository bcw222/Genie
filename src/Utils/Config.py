import logging
import os
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)
CONFIG_FILE_PATH = "./Config.txt"


@dataclass
class Config:
    # --- 设备配置 (使用 onnxruntime 自动检测) ---
    HUBERT_MODEL_PATH: str = "./Data/chinese-hubert-base.onnx"
    MAX_CACHED_CHARACTER_MODELS: int = 3
    MAX_CACHED_REFERENCE_AUDIO: int = 5


def _initialize_config() -> Config:
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                instance = Config(**yaml_data)
                logger.info(f"成功从 {CONFIG_FILE_PATH} 加载配置。")
                return instance
        except (yaml.YAMLError, TypeError, KeyError) as e:
            logger.warning(
                f"无法解析配置文件 {CONFIG_FILE_PATH} (错误: {e})，"
                "正在使用默认配置。"
            )
            return Config()
    else:
        logger.info(f"配置文件 {CONFIG_FILE_PATH} 未找到，将使用默认配置并创建新文件。")
        default_instance = Config()
        try:
            with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(default_instance), f, allow_unicode=True, default_flow_style=False)  # type: ignore
        except IOError as e:
            logger.error(f"无法创建或写入配置文件 {CONFIG_FILE_PATH}: {e}")

        return default_instance


config: Config = _initialize_config()
