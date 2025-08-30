import os, asyncio
MODEL_NAME = os.getenv("MODEL_NAME", "stub-llm")

# NOTE: This is a dev-friendly stub. Swap with HF Transformers or vLLM later.
async def generate_text(prompt: str, max_tokens: int, temperature: float):
    await asyncio.sleep(0.05)  # simulate work
    # Trivial "generation": echo with marker (replace with real model invocation)
    text = f"[{MODEL_NAME}] {prompt[:128]} ... (generated)"
    tokens = min(max_tokens, max(8, len(prompt)//4))
    return text, tokens, MODEL_NAME
