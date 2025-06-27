import argparse
import csv
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example usage:
# CUDA_VISIBLE_DEVICES=0 python -m utils.dataset_alpaca --input_csv dataset/alpaca_instructions_100.csv --output_csv dataset/alpaca_reasoning_template.csv


def parse_args():
    parser = argparse.ArgumentParser(description="Process prompts through DeepSeek-R1-Distill-Llama-8B model")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file containing prompts")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file for results")
    parser.add_argument("--prompt_column", type=str, default="instruction", help="Column name containing prompts")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    return parser.parse_args()


def load_model(device):
    """Load the DeepSeek-R1-Distill-Llama-8B model and tokenizer"""
    print("Loading DeepSeek-R1-Distill-Llama-8B model and tokenizer...")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def read_prompts(input_csv, prompt_column):
    """Read prompts from the input CSV file"""
    prompts = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if prompt_column not in reader.fieldnames:
            raise ValueError(f"Column '{prompt_column}' not found in CSV. Available columns: {reader.fieldnames}")
        for row in reader:
            prompts.append(row[prompt_column])
    return prompts


def process_prompts(model, tokenizer, prompts, max_new_tokens, device):
    """Process prompts through the model and return outputs"""
    results = []
    for prompt in tqdm(prompts):
        chat = [{"role": "user", "content": prompt}]
        # Get tokenized chat format without converting to tensor yet
        raw_tokenized_chat = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        
        # Print the decoded tokenized chat for debugging
        decoded_chat = tokenizer.decode(raw_tokenized_chat)
        print("="*50)
        print("DECODED CHAT:")
        print(decoded_chat)
        print("="*50)
        
        # Now convert to tensor for model input
        tokenized_chat = torch.tensor([raw_tokenized_chat]).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                tokenized_chat,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Use greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated output
            response = tokenizer.decode(output_ids[0][len(raw_tokenized_chat):], skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "response": response
            })
    return results


def save_results(results, output_csv):
    """Save input-output pairs to the output CSV file"""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_csv}")


def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    # Load model and tokenizer
    model, tokenizer = load_model(args.device)
    # Read prompts from input CSV
    prompts = read_prompts(args.input_csv, args.prompt_column)
    print(f"Loaded {len(prompts)} prompts from {args.input_csv}")
    # Process prompts through the model
    results = process_prompts(model, tokenizer, prompts, args.max_new_tokens, args.device)
    # Save results to output CSV
    save_results(results, args.output_csv)


def run():
    main()


if __name__ == "__main__":
    main()