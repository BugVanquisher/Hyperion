# Example implementation with HuggingFace
import asyncio
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "auto")  # auto, cpu, cuda
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
ENABLE_QUANTIZATION = os.getenv("ENABLE_QUANTIZATION", "false").lower() == "true"
ENABLE_OPTIMIZATION = os.getenv("ENABLE_OPTIMIZATION", "true").lower() == "true"

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
            logger.info(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            )
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


def optimize_model(model, device):
    """Apply various optimizations to the model."""
    if not ENABLE_OPTIMIZATION:
        return model

    logger.info("Applying model optimizations...")

    # Quantization for CPU inference
    if device.type == "cpu" and ENABLE_QUANTIZATION:
        try:
            logger.info("Applying dynamic quantization for CPU...")
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8  # Quantize linear layers
            )
            logger.info("Dynamic quantization applied successfully")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed, using original model: {e}")

    # GPU optimizations
    if device.type == "cuda":
        try:
            # Enable optimized attention (if available)
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                logger.info("Using optimized scaled dot product attention")

            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, "compile"):
                logger.info("Compiling model with torch.compile...")
                compiled_model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compilation completed")
                return compiled_model

        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")

    # Memory optimization
    try:
        # Enable memory efficient attention if available
        if hasattr(model.config, "use_memory_efficient_attention"):
            model.config.use_memory_efficient_attention = True
            logger.info("Enabled memory efficient attention")
    except Exception as e:
        logger.debug(f"Memory optimization not available: {e}")

    return model


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
            low_cpu_mem_usage=True,
        )

        # Move model to device
        model = model.to(device)

        # Apply optimizations
        model = optimize_model(model, device)

        # Enable optimization for inference
        model.eval()

        logger.info(f"Model loaded successfully on {device}")

        # Log model info
        if hasattr(model, "config"):
            logger.info(f"Model parameters: ~{model.num_parameters() / 1e6:.1f}M")

        if device.type == "cuda":
            logger.info(
                f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
            )
            logger.info(
                f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            )


async def generate_text(prompt: str, max_tokens: int, temperature: float):
    """Generate text for a single prompt. For batch processing, use generate_text_batch."""
    results = await generate_text_batch([prompt], [max_tokens], [temperature])
    response, tokens, model_name = results[0]
    return response, tokens, model_name


async def generate_text_batch(
    prompts: list[str], max_tokens_list: list[int], temperatures: list[float]
):
    """
    Generate text for multiple prompts using true batch inference.

    Args:
        prompts: List of input prompts
        max_tokens_list: List of max_tokens for each prompt
        temperatures: List of temperatures for each prompt

    Returns:
        List of tuples (response, tokens_used, model_name) for each prompt
    """
    await init_model()

    # For true batching, we need to handle different generation parameters
    # Group requests by identical parameters for efficient batching
    param_groups = {}
    for i, (prompt, max_tok, temp) in enumerate(
        zip(prompts, max_tokens_list, temperatures)
    ):
        key = (max_tok, temp)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append((i, prompt))

    # Process each parameter group as a batch
    results = [None] * len(prompts)

    for (max_tok, temp), indexed_prompts in param_groups.items():
        indices = [idx for idx, _ in indexed_prompts]
        batch_prompts = [prompt for _, prompt in indexed_prompts]

        # Tokenize all prompts in the batch
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # True batch generation
        with torch.no_grad():
            if device.type == "cuda":
                # GPU optimized batch generation
                with torch.cuda.amp.autocast():
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + max_tok,
                        temperature=temp,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        early_stopping=True,
                    )
            else:
                # CPU batch generation
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_tok,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

        # Decode each response in the batch
        for i, (idx, output) in enumerate(zip(indices, outputs)):
            input_length = input_ids[i].shape[0]
            response = tokenizer.decode(
                output[input_length:], skip_special_tokens=True
            )
            tokens_generated = len(output) - input_length
            results[idx] = (response.strip(), tokens_generated, MODEL_NAME)

    return results


async def health_check():
    """Check if the model and tokenizer are properly loaded."""
    return model is not None and tokenizer is not None


def get_device_info():
    """Get detailed device information for monitoring."""
    if device is None:
        return {"device": "not_initialized"}

    info = {
        "device_type": device.type,
        "device_name": str(device),
        "optimizations": {
            "quantization_enabled": ENABLE_QUANTIZATION,
            "optimization_enabled": ENABLE_OPTIMIZATION,
            "batch_size": BATCH_SIZE,
        },
    }

    if device.type == "cuda" and torch.cuda.is_available():
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f}GB",
                "cuda_version": torch.version.cuda,
                "torch_compile_available": hasattr(torch, "compile"),
            }
        )

    # Model optimization info
    if model is not None:
        info["model_info"] = {
            "model_name": MODEL_NAME,
            "model_type": type(model).__name__,
            "is_quantized": "quantized" in str(type(model)).lower(),
            "parameters": (
                getattr(model, "num_parameters", lambda: 0)()
                if hasattr(model, "num_parameters")
                else "unknown"
            ),
        }

    return info
