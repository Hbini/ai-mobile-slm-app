"""Unit tests for ModelManager module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from src.model_manager import ModelManager, ModelConfig, ModelRegistry


class TestModelConfig:
    """Test ModelConfig initialization and validation."""

    def test_model_config_initialization(self):
        """Test creating a ModelConfig instance."""
        config = ModelConfig(
            model_name="gemma-2b",
            model_type="slm",
            quantization="int8",
            device="cpu"
        )
        assert config.model_name == "gemma-2b"
        assert config.model_type == "slm"
        assert config.quantization == "int8"
        assert config.device == "cpu"

    def test_model_config_with_invalid_device(self):
        """Test ModelConfig validation for invalid device."""
        with pytest.raises(ValueError):
            ModelConfig(
                model_name="gemma-2b",
                model_type="slm",
                device="invalid_device"
            )


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_register_model(self):
        """Test registering a model."""
        registry = ModelRegistry()
        config = ModelConfig(
            model_name="test-model",
            model_type="slm"
        )
        registry.register("test-model", config)
        assert registry.get("test-model") == config

    def test_get_nonexistent_model(self):
        """Test retrieving non-existent model raises error."""
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_models(self):
        """Test listing all registered models."""
        registry = ModelRegistry()
        config1 = ModelConfig(model_name="model1", model_type="slm")
        config2 = ModelConfig(model_name="model2", model_type="llm")
        registry.register("model1", config1)
        registry.register("model2", config2)
        models = registry.list_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models


class TestModelManager:
    """Test ModelManager core functionality."""

    @pytest.fixture
    def manager(self):
        """Create a ModelManager instance for testing."""
        return ModelManager()

    @patch('torch.cuda.is_available')
    def test_initialization_with_cuda(self, mock_cuda):
        """Test initialization with CUDA available."""
        mock_cuda.return_value = True
        manager = ModelManager()
        assert manager.device in ["cuda", "cpu"]

    def test_load_model(self, manager):
        """Test loading a model."""
        config = ModelConfig(
            model_name="test-model",
            model_type="slm",
            device="cpu"
        )
        with patch.object(manager, '_load_from_cache') as mock_load:
            mock_load.return_value = MagicMock()
            manager.load_model(config)
            mock_load.assert_called_once()

    def test_load_model_with_quantization(self, manager):
        """Test loading a quantized model."""
        config = ModelConfig(
            model_name="test-model",
            model_type="slm",
            quantization="int8",
            device="cpu"
        )
        with patch.object(manager, '_apply_quantization') as mock_quant:
            manager.load_model(config)
            mock_quant.assert_called_once()

    def test_inference(self, manager):
        """Test model inference."""
        manager.model = MagicMock()
        manager.model.return_value = torch.tensor([[0.1, 0.2, 0.3]])
        
        result = manager.inference("Test input")
        assert result is not None
        manager.model.assert_called_once()

    def test_memory_usage(self, manager):
        """Test memory usage tracking."""
        memory = manager.get_memory_usage()
        assert isinstance(memory, dict)
        assert 'allocated' in memory
        assert 'reserved' in memory

    def test_model_unload(self, manager):
        """Test unloading model and freeing memory."""
        manager.model = MagicMock()
        manager.unload_model()
        assert manager.model is None


class TestEdgeOptimization:
    """Test edge optimization features."""

    def test_model_pruning(self):
        """Test model pruning for edge deployment."""
        from src.model_manager import prune_model
        mock_model = MagicMock()
        pruned = prune_model(mock_model, sparsity=0.3)
        assert pruned is not None

    def test_model_distillation(self):
        """Test model distillation."""
        from src.model_manager import distill_model
        teacher_model = MagicMock()
        student_config = ModelConfig(
            model_name="student",
            model_type="slm"
        )
        distilled = distill_model(teacher_model, student_config)
        assert distilled is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_model_type(self):
        """Test handling of invalid model types."""
        with pytest.raises(ValueError):
            ModelConfig(
                model_name="test",
                model_type="invalid_type"
            )

    def test_model_load_failure(self):
        """Test handling of model load failures."""
        manager = ModelManager()
        config = ModelConfig(
            model_name="nonexistent-model",
            model_type="slm"
        )
        with pytest.raises(Exception):
            manager.load_model(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
