"""
Integration tests for ML model functionality.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest
import torch

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app.models.llm import (
    generate_text,
    get_device_info,
    get_optimal_device,
    health_check,
    init_model,
)


class TestModelInitialization:
    """Test model initialization and device selection."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global model state before each test."""
        # Import the module to reset globals
        import app.models.llm as llm_module

        llm_module.model = None
        llm_module.tokenizer = None
        llm_module.device = None

    @patch.dict(os.environ, {"DEVICE_TYPE": "cpu"})
    def test_get_optimal_device_cpu_forced(self):
        """Test device selection when CPU is forced."""
        device = get_optimal_device()
        assert device.type == "cpu"

    @patch.dict(os.environ, {"DEVICE_TYPE": "cuda"})
    @patch("torch.cuda.is_available", return_value=False)
    def test_get_optimal_device_cuda_fallback(self):
        """Test CUDA fallback to CPU when CUDA unavailable."""
        device = get_optimal_device()
        assert device.type == "cpu"

    @patch.dict(os.environ, {"DEVICE_TYPE": "auto"})
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="Test GPU")
    @patch("torch.cuda.get_device_properties")
    def test_get_optimal_device_auto_cuda(self, mock_props):
        """Test auto device selection with CUDA available."""
        mock_props.return_value.total_memory = 8e9  # 8GB
        device = get_optimal_device()
        assert device.type == "cuda"

    def test_get_device_info(self):
        """Test device information retrieval."""
        info = get_device_info()
        assert "device_type" in info
        assert "available_devices" in info
        assert isinstance(info["available_devices"], list)

    @patch("app.models.llm.AutoTokenizer.from_pretrained")
    @patch("app.models.llm.AutoModelForCausalLM.from_pretrained")
    def test_init_model_success(self, mock_model, mock_tokenizer):
        """Test successful model initialization."""
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        result = init_model()

        assert result is True
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()

    @patch("app.models.llm.AutoTokenizer.from_pretrained")
    def test_init_model_failure(self, mock_tokenizer):
        """Test model initialization failure handling."""
        mock_tokenizer.side_effect = Exception("Model not found")

        result = init_model()

        assert result is False

    def test_health_check_no_model(self):
        """Test health check when model is not loaded."""
        status = health_check()
        assert status["model_loaded"] is False
        assert "error" in status

    @patch("app.models.llm.model", Mock())
    @patch("app.models.llm.tokenizer", Mock())
    def test_health_check_with_model(self):
        """Test health check when model is loaded."""
        status = health_check()
        assert status["model_loaded"] is True
        assert "error" not in status


class TestTextGeneration:
    """Test text generation functionality."""

    @pytest.fixture
    def mock_model_loaded(self):
        """Fixture to mock a loaded model."""
        with patch("app.models.llm.model") as mock_model, patch(
            "app.models.llm.tokenizer"
        ) as mock_tokenizer, patch("app.models.llm.device", torch.device("cpu")):

            # Setup tokenizer mock
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "Generated response"
            mock_tokenizer.pad_token = "[PAD]"

            # Setup model mock
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

            yield mock_model, mock_tokenizer

    def test_generate_text_no_model(self):
        """Test text generation when model is not loaded."""
        result = generate_text("Hello", max_tokens=10, temperature=0.7)

        assert result["error"] == "Model not loaded"
        assert result["response"] == ""

    def test_generate_text_success(self, mock_model_loaded):
        """Test successful text generation."""
        mock_model, mock_tokenizer = mock_model_loaded

        result = generate_text("Hello", max_tokens=10, temperature=0.7)

        assert result["error"] is None
        assert result["response"] == "Generated response"
        assert result["tokens_used"] == 5
        assert "processing_time_ms" in result

        # Verify tokenizer calls
        mock_tokenizer.encode.assert_called()
        mock_tokenizer.decode.assert_called()

        # Verify model generate call
        mock_model.generate.assert_called()

    def test_generate_text_with_different_params(self, mock_model_loaded):
        """Test text generation with different parameters."""
        mock_model, mock_tokenizer = mock_model_loaded

        result = generate_text("Test prompt", max_tokens=50, temperature=0.9)

        assert result["error"] is None
        # Verify model was called with correct parameters
        call_args = mock_model.generate.call_args[1]
        assert call_args["max_new_tokens"] == 50
        assert abs(call_args["temperature"] - 0.9) < 0.01

    def test_generate_text_exception_handling(self, mock_model_loaded):
        """Test exception handling during text generation."""
        mock_model, mock_tokenizer = mock_model_loaded
        mock_model.generate.side_effect = RuntimeError("GPU out of memory")

        result = generate_text("Hello", max_tokens=10, temperature=0.7)

        assert "error" in result
        assert "GPU out of memory" in result["error"]
        assert result["response"] == ""


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests that require actual model loading (slow tests)."""

    @pytest.mark.slow
    def test_full_model_pipeline(self):
        """Test the complete model pipeline from init to generation."""
        # This test requires actual model files and can be slow
        # Mark it as slow so it can be skipped in fast test runs

        # Try to initialize the model
        success = init_model()

        if not success:
            pytest.skip("Model initialization failed - may need model files")

        # Test health check
        health = health_check()
        assert health["model_loaded"] is True

        # Test generation
        result = generate_text("Hello", max_tokens=5, temperature=0.7)
        assert result["error"] is None
        assert len(result["response"]) > 0
        assert result["tokens_used"] > 0


@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for model operations."""

    @patch("app.models.llm.model", Mock())
    @patch("app.models.llm.tokenizer", Mock())
    @patch("app.models.llm.device", torch.device("cpu"))
    def test_generation_performance(self):
        """Test that text generation completes within reasonable time."""
        import time

        start_time = time.time()
        result = generate_text("Test prompt", max_tokens=10, temperature=0.7)
        duration = time.time() - start_time

        # Should complete within 5 seconds even on CPU
        assert duration < 5.0
        assert "processing_time_ms" in result
