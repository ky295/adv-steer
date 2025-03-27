import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import csv


# Example usage:
# CUDA_VISIBLE_DEVICES=0 python -m utils.dataset_strong_reject \
#   --output_csv dataset/strongreject_reasoning_8B.csv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through DeepSeek-R1-Distill-Llama-8B model"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file for results"
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--append_words", type=str, default="Please begin all of your responses with \"<think>\\n\" followed by your step-by-step reasoning process, and then provide your final answer.", help="Words to append to each prompt")
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


def read_prompts():
    """Read prompts from the HuggingFace dataset"""
    strongreject_dataset = load_dataset("csv", data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv")
    train_dataset = strongreject_dataset['train']
    
    if "forbidden_prompt" not in train_dataset.column_names:
        raise ValueError(f"Column not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["forbidden_prompt"]
    return prompts


def process_prompts(model, tokenizer, prompts, max_new_tokens, device, batch_size, append_words):
    """Process prompts through the model and return outputs"""
    results = []
    # Append the word to each prompt
    modified_prompts = [f"{prompt} {append_words}" for prompt in prompts]
    # Process in batches
    for i in tqdm(range(0, len(modified_prompts), batch_size), desc="Processing prompts"):
        batch_prompts = modified_prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Use greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
        # Decode the generated outputs
        for j, output_id in enumerate(output_ids):
            prompt_tokens_length = inputs.input_ids[j].shape[0]
            response_ids = output_id[prompt_tokens_length:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            results.append({
                "prompt": prompts[i+j],  # Store the original prompt without appended word
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
    # Read prompts from HuggingFace dataset
    prompts = read_prompts()
    print(f"Loaded {len(prompts)} prompts from strong_reject dataset")
    # Process prompts through the model
    results = process_prompts(model, tokenizer, prompts, args.max_new_tokens, args.device, args.batch_size, args.append_words)
    # Save results to output CSV
    save_results(results, args.output_csv)


def run():
    main()


if __name__ == "__main__":
    main()