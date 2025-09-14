# Example implementation with HuggingFace
import os
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
model = None
tokenizer = None

async def init_model():
    global model, tokenizer
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        if torch.cuda.is_available():
            model = model.cuda()

async def generate_text(prompt: str, max_tokens: int, temperature: float):
    await init_model()
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=inputs.shape[1] + max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip(), len(outputs[0]) - inputs.shape[1], MODEL_NAME

async def health_check():
    """Check if the model and tokenizer are properly loaded."""
    global model, tokenizer
    return model is not None and tokenizer is not None