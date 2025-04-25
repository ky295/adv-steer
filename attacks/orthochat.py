import os
# os.environ["HF_HOME"] = "/data/math-lasr/shil6499/huggingface"
os.environ["HF_HOME"] = "/scratch/etheridge/huggingface" # move to .env / ~/.profile

# Import argparse for command-line arguments
import argparse
import torch
import re
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

ORTHO_MODEL = "kureha295/ortho_model" 
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Default to using one GPU
DEFAULT_DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 2048+1024

# Colors
BLUE = "\033[34m"
PINK = "\033[95m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"

def parse_args():
    parser = argparse.ArgumentParser(description="Compare ORTHO and BASE model outputs side by side")
    parser.add_argument("--ortho-gpu", type=str, default=DEFAULT_DEVICE, 
                        help=f"GPU device for ORTHO model (e.g., 'cuda:0'). Default: {DEFAULT_DEVICE}")
    parser.add_argument("--base-gpu", type=str, default=DEFAULT_DEVICE, 
                        help=f"GPU device for BASE model (e.g., 'cuda:1'). Default: {DEFAULT_DEVICE}")
    return parser.parse_args()

def load_model(model_name, device):
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
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
    hardcoded_special_tokens = ["<think>", "</think>"] # not being recognised?
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
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_output = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    
    # Split into thinking and answer parts using regex
    think_pattern = r"(.*?)</think>"
    thinking_match = re.search(think_pattern, raw_output, re.DOTALL)
    
    thinking = thinking_match.group(1).strip() if thinking_match else ""
    # Get everything after </think>
    answer = re.sub(think_pattern, "", raw_output, flags=re.DOTALL).strip()
    
    return {
        "thinking": thinking,
        "answer": answer
    }

def display_side_by_side(base_response, ortho_response):
    """Display responses with thinking and answer sections separated."""

    def split_text_to_lines(text, max_width):
        """Split text into lines that fit within max_width."""
        lines = []
        for paragraph in text.split('\n'):
            if len(paragraph) <= max_width:
                lines.append(paragraph)
            else:
                # Split long lines
                current_line = ""
                for word in paragraph.split(' '):
                    if len(current_line) + len(word) + 1 <= max_width:
                        current_line += (word + " ")
                    else:
                        lines.append(current_line.rstrip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.rstrip())
        return lines

    terminal_width = shutil.get_terminal_size().columns
    col_width = (terminal_width // 2) - 1
    divider = "─" * terminal_width
    
    print(divider)
    left_header_text = f"😇 {BASE_MODEL} 😇"
    right_header_text = f"👹 {ORTHO_MODEL} 👹"
    left_header = f"{BOLD}{YELLOW}{left_header_text}{RESET}"
    right_header = f"{BOLD}{GREEN}{right_header_text}{RESET}"
    
    left_padding = (col_width - len(left_header_text)) // 2
    right_padding = (col_width - len(right_header_text)) // 2
    print(f"{' ' * left_padding}{left_header}{' ' * (col_width - left_padding - len(left_header_text))} │ {' ' * right_padding}{right_header}")
    print(divider)
    
    thinking_header = f"{BOLD}THINKING{RESET}"
    thinking_padding = (terminal_width - len("THINKING")) // 2
    print(f"{' ' * thinking_padding}{thinking_header}")
    print(divider)
    
    base_thinking_lines = split_text_to_lines(f"{base_response['thinking']}", col_width)
    ortho_thinking_lines = split_text_to_lines(f"{ortho_response['thinking']}", col_width)
    
    max_lines = max(len(base_thinking_lines), len(ortho_thinking_lines))
    base_thinking_lines += [""] * (max_lines - len(base_thinking_lines))
    ortho_thinking_lines += [""] * (max_lines - len(ortho_thinking_lines))
    
    for base, ortho in zip(base_thinking_lines, ortho_thinking_lines):
        base_formatted = base.ljust(col_width)
        print(f"{base_formatted} │ {ortho}")
    
    print(divider)
    
    answer_header = f"{BOLD}ANSWER{RESET}"
    answer_padding = (terminal_width - len("ANSWER")) // 2
    print(f"{' ' * answer_padding}{answer_header}")
    
    base_answer_lines = split_text_to_lines(base_response["answer"], col_width)
    ortho_answer_lines = split_text_to_lines(ortho_response["answer"], col_width)
    
    max_lines = max(len(base_answer_lines), len(ortho_answer_lines))
    base_answer_lines += [""] * (max_lines - len(base_answer_lines))
    ortho_answer_lines += [""] * (max_lines - len(ortho_answer_lines))
    
    for base, ortho in zip(base_answer_lines, ortho_answer_lines):
        base_formatted = base.ljust(col_width)
        print(f"{base_formatted} │ {ortho}")
    
    print(divider+"\n\n")



def main():
    args = parse_args()
    
    # Load both models on specified GPUs
    base_model, base_tokenizer = load_model(BASE_MODEL, args.base_gpu)
    ortho_model, ortho_tokenizer = load_model(ORTHO_MODEL, args.ortho_gpu)
    
    print(f"\nReady to chat! Press Ctrl+C to exit.")
    print(f"BASE model on {args.base_gpu}, ORTHO model on {args.ortho_gpu}")
    print("(Note: there is no chat history - each message has no prior context)\n")

    try:
        while True:
            user_input = input(f"{PINK}You:{RESET} ")
            
            # Process the input with both models in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Prepare inputs
                base_ids, base_len = apply_chat_template(user_input, base_tokenizer, args.base_gpu)
                ortho_ids, ortho_len = apply_chat_template(user_input, ortho_tokenizer, args.ortho_gpu)
                
                # Submit generation tasks
                base_future = executor.submit(generate_response, base_model, base_tokenizer, base_ids, base_len, MAX_NEW_TOKENS)
                ortho_future = executor.submit(generate_response, ortho_model, ortho_tokenizer, ortho_ids, ortho_len, MAX_NEW_TOKENS)
                
                # Get results
                base_output = base_future.result()
                ortho_output = ortho_future.result()
            
            # Display responses side by side (BASE on left, ORTHO on right)
            display_side_by_side(base_output, ortho_output)
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()