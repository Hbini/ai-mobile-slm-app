"""Model Manager for Edge AI - Handles quantization and lazy loading.

This module provides the core infrastructure for managing small language models
on mobile devices with limited memory. It implements lazy loading, memory pressure
detection, and dynamic quantization based on device capabilities.
"""

from typing import Dict, Optional, Tuple
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class DeviceInfo:
    """Information about target device capabilities."""
    release_year: int
    available_ram_gb: float
    cpu_cores: int
    device_name: str


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: str
    model_size_mb: float
    quantization: str
    is_critical: bool = False
    last_access_time: float = field(default_factory=time.time)


class MemoryMonitor:
    """Monitors device memory pressure and triggers cleanup."""
    
    def __init__(self, device_ram_mb: int):
        self.device_ram_mb = device_ram_mb
        self.pressure_threshold = 0.8
    
    def get_pressure(self) -> float:
        """Get current memory pressure as ratio (0.0-1.0)."""
        # In real implementation, check actual system memory
        # For demo: simulate based on number of models
        return min(1.0, 0.5)  # Placeholder
    
    def is_memory_critical(self) -> bool:
        """Check if memory pressure exceeds threshold."""
        return self.get_pressure() > self.pressure_threshold


class QuantizationManager:
    """Manages model quantization strategies."""
    
    QUANTIZATION_PROFILES = {
        "4bit": {"memory_reduction": 0.75, "accuracy_loss": 0.08},
        "8bit": {"memory_reduction": 0.5, "accuracy_loss": 0.02},
        "16bit": {"memory_reduction": 0.0, "accuracy_loss": 0.0},
    }
    
    @staticmethod
    def select_quantization(device_info: DeviceInfo) -> str:
        """Choose optimal quantization level based on device."""
        if device_info.release_year < 2020 or device_info.available_ram_gb < 2:
            return "4bit"
        elif device_info.release_year < 2022 or device_info.available_ram_gb < 4:
            return "8bit"
        else:
            return "16bit"
    
    @staticmethod
    def apply_quantization(model: str, method: str) -> str:
        """Apply quantization to model weights."""
        return f"{model}_{method}"


class ModelManager:
    """Manages model lifecycle: loading, unloading, and memory optimization."""
    
    def __init__(self, device_info: DeviceInfo):
        self.device_info = device_info
        self.loaded_models: Dict[str, object] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.memory_monitor = MemoryMonitor(int(device_info.available_ram_gb * 1024))
        self.quantization_manager = QuantizationManager()
    
    def load_model(self, model_name: str, force: bool = False) -> Optional[object]:
        """Lazy load model on demand with memory pressure checks."""
        if model_name in self.loaded_models and not force:
            self._update_access_time(model_name)
            return self.loaded_models[model_name]
        
        # Check memory pressure
        if self.memory_monitor.is_memory_critical() and not force:
            self._unload_least_used_model()
        
        # Determine quantization
        quantization = QuantizationManager.select_quantization(self.device_info)
        
        # Simulate loading (in real app: load actual model)
        model = f"<Model:{model_name} Quantization:{quantization}>"
        self.loaded_models[model_name] = model
        self.model_configs[model_name] = ModelConfig(
            model_id=model_name,
            model_size_mb=100.0,  # Simulated size
            quantization=quantization
        )
        
        return model
    
    def _update_access_time(self, model_name: str) -> None:
        """Update last access time for LRU tracking."""
        if model_name in self.model_configs:
            self.model_configs[model_name].last_access_time = time.time()
    
    def _unload_least_used_model(self) -> None:
        """Remove least recently used model when memory pressure detected."""
        if not self.loaded_models:
            return
        
        # Find LRU non-critical model
        lru_model = min(
            (
                (name, cfg) for name, cfg in self.model_configs.items()
                if not cfg.is_critical and name in self.loaded_models
            ),
            key=lambda x: x[1].last_access_time,
            default=(None, None)
        )
        
        if lru_model[0]:
            del self.loaded_models[lru_model[0]]
            print(f"Unloaded model: {lru_model[0]}")
    
    def get_model_memory_usage(self, model_name: str) -> float:
        """Get estimated memory usage for a model."""
        if model_name not in self.model_configs:
            return 0.0
        
        config = self.model_configs[model_name]
        reduction = QuantizationManager.QUANTIZATION_PROFILES[
            config.quantization
        ]["memory_reduction"]
        return config.model_size_mb * (1 - reduction)
    
    def unload_all(self) -> None:
        """Clear all loaded models."""
        self.loaded_models.clear()


if __name__ == "__main__":
    # Example usage
    device = DeviceInfo(
        release_year=2021,
        available_ram_gb=4,
        cpu_cores=8,
        device_name="Pixel 6"
    )
    
    manager = ModelManager(device)
    model1 = manager.load_model("phi-2")
    model2 = manager.load_model("mistral-7b")
    
    print(f"Loaded: {model1}")
    print(f"Loaded: {model2}")
    print(f"Memory: {manager.get_model_memory_usage('phi-2'):.1f}MB")
