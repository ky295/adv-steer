import os
# os.environ["HF_HOME"] = "/data/math-lasr/shil6499/huggingface"
os.environ["HF_HOME"] = "/scratch/etheridge/huggingface" # move to .env / ~/.profile

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "kureha295/ortho_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 2048+1024

BLUE = "\033[34m"
PINK = "\033[95m"
RED = "\033[31m"
RESET = "\033[0m"

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer

def apply_chat_template(prompt, tokenizer, device):
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(device)
    return input_ids, input_ids.shape[-1]

def highlight_special_tokens(text, tokenizer):
    # Get all special tokens from tokenizer
    special_tokens = []
    for attr in dir(tokenizer):
        if attr.endswith('_token') and hasattr(tokenizer, attr):
            token = getattr(tokenizer, attr)
            if isinstance(token, str) and token:
                special_tokens.append(token)
    
    # Add additional common special tokens if they exist in tokenizer
    for token in tokenizer.all_special_tokens:
        if token not in special_tokens:
            special_tokens.append(token)
    hardcoded_special_tokens = ["<think>", "</think>"] # not being recognised?
    special_tokens.extend(hardcoded_special_tokens)

    # Escape special regex characters and sort by length (longest first to avoid partial matches)
    special_tokens = sorted([re.escape(token) for token in special_tokens if token], key=len, reverse=True)
    
    # Highlight each special token in the text
    for token in special_tokens:
        if token in text:
            text = text.replace(token, f"{RED}{token}{RESET}")
    
    return text

def generate_response(model, tokenizer, input_ids, input_len, max_new_tokens):
    # Create attention mask (all 1s for the input tokens)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Enable sampling
            temperature=0.6,  # Keep temperature
            top_p=0.95,      # Keep top_p
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_output = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    # Highlight special tokens in the output
    highlighted_output = highlight_special_tokens("\n<think>\n"+raw_output, tokenizer)
    return highlighted_output

def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME, DEVICE)
    print("Ready to chat! Press Ctrl+C to exit.\n")

    try:
        while True:
            user_input = input(f"{PINK}You:{RESET} ")
            
            # Process the input - TODO: maybe include chat history?
            input_ids, input_len = apply_chat_template(user_input, tokenizer, DEVICE)
            output = generate_response(model, tokenizer, input_ids, input_len, MAX_NEW_TOKENS)
            
            # Print model response with blue label
            print(f"{BLUE}Model:{RESET} {output}\n")
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()