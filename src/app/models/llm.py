# Example implementation with HuggingFace
import os
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up logging
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "auto")  # auto, cpu, cuda
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))

model = None
tokenizer = None
device = None

def get_optimal_device():
    """Detect and configure the optimal device for inference."""
    global device

    if device is not None:
        return device

    if DEVICE_TYPE == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU device (forced)")
    elif DEVICE_TYPE == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Auto-detected CUDA device: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = torch.device("cpu")
            logger.info("Auto-detected CPU device (CUDA not available)")

    return device

async def init_model():
    global model, tokenizer, device
    if model is None:
        logger.info(f"Loading model: {MODEL_NAME}")

        # Configure device
        device = get_optimal_device()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load model with device configuration
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )

        # Move model to device
        model = model.to(device)

        # Enable optimization for inference
        model.eval()

        logger.info(f"Model loaded successfully on {device}")

        # Log model info
        if hasattr(model, 'config'):
            logger.info(f"Model parameters: ~{model.num_parameters() / 1e6:.1f}M")

        if device.type == "cuda":
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

async def generate_text(prompt: str, max_tokens: int, temperature: float):
    await init_model()

    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate with optimized settings
    with torch.no_grad():
        if device.type == "cuda":
            # GPU optimized generation
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    early_stopping=True
                )
        else:
            # CPU generation
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

    # Decode response
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip(), len(outputs[0]) - inputs.shape[1], MODEL_NAME

async def health_check():
    """Check if the model and tokenizer are properly loaded."""
    global model, tokenizer
    return model is not None and tokenizer is not None

def get_device_info():
    """Get detailed device information for monitoring."""
    if device is None:
        return {"device": "not_initialized"}

    info = {
        "device_type": device.type,
        "device_name": str(device)
    }

    if device.type == "cuda" and torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f}GB",
            "cuda_version": torch.version.cuda
        })

    return info