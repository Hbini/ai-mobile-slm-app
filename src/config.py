"""Configuration and settings for AI Mobile SLM App."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DeviceConfig:
    """Device configuration settings."""
    device_type: str = "cpu"  # cpu, cuda, tpu
    memory_limit_mb: int = 2048
    enable_gpu: bool = False
    
    def validate(self) -> bool:
        """Validate device configuration."""
        valid_types = ['cpu', 'cuda', 'tpu']
        return self.device_type in valid_types and self.memory_limit_mb > 0


@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_name: str = "gemma-2b"
    model_type: str = "slm"
    quantization: str = "int8"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    def validate(self) -> bool:
        """Validate model configuration."""
        valid_types = ['slm', 'llm']
        valid_quant = ['int8', 'int4', 'fp16', 'fp32']
        return (self.model_type in valid_types and 
                self.quantization in valid_quant and
                0 <= self.temperature <= 1.0)


@dataclass
class StorageConfig:
    """Storage configuration settings."""
    cache_dir: Path = Path("/tmp/ai_mobile_cache")
    model_cache_dir: Path = Path("/tmp/models")
    database_url: str = "sqlite:///./app.db"
    enable_offline_mode: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = "AI Mobile SLM App"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    
    device_config: DeviceConfig = None
    model_config: ModelConfig = None
    storage_config: StorageConfig = None
    
    def __post_init__(self):
        """Initialize nested configs."""
        if self.device_config is None:
            self.device_config = DeviceConfig()
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.storage_config is None:
            self.storage_config = StorageConfig()
    
    def validate(self) -> bool:
        """Validate all configurations."""
        return (self.device_config.validate() and
                self.model_config.validate())
    
    @staticmethod
    def from_env() -> 'AppConfig':
        """Load configuration from environment variables."""
        return AppConfig(
            app_name=os.getenv('APP_NAME', 'AI Mobile SLM App'),
            version=os.getenv('APP_VERSION', '0.1.0'),
            debug=os.getenv('DEBUG', 'False').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            device_config=DeviceConfig(
                device_type=os.getenv('DEVICE_TYPE', 'cpu'),
                memory_limit_mb=int(os.getenv('MEMORY_LIMIT_MB', 2048)),
                enable_gpu=os.getenv('ENABLE_GPU', 'False').lower() == 'true'
            )
        )


# Global configuration instance
config = AppConfig.from_env()
