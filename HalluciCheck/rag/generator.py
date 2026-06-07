import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model(device="auto"):
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype     = torch.float16 if device != "cpu" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = dtype,
        device_map  = device
    )
    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, tokenizer


def generate_baseline(question, model, tokenizer, max_new_tokens=150):
    prompt = f"""<|system|>
You are a factual assistant. Answer the question truthfully and concisely in 1-2 sentences.</s>
<|user|>
{question}</s>
<|assistant|>"""

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature    = 0.7,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id
        )
    latency = round(time.time() - start, 2)

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer      = full_output.split("<|assistant|>")[-1].strip()
    return answer, latency


def generate_rag(question, context, model, tokenizer, max_new_tokens=150):
    prompt = f"""<|system|>
You are a factual assistant. Use the context below to answer the question accurately and concisely.</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>"""

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature    = 0.7,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id
        )
    latency = round(time.time() - start, 2)

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer      = full_output.split("<|assistant|>")[-1].strip()
    return answer, latency