import argparse
import csv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# example use python utils/dataset.py --input_csv dataset/alpaca_instructions_100.csv --output_csv dataset/alpaca_reasoning.csv

def parse_args():
    parser = argparse.ArgumentParser(description="Process prompts through DeepSeek-R1-Distill-Llama-8B model")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file containing prompts")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file for results")
    parser.add_argument("--prompt_column", type=str, default="instruction", help="Column name containing prompts")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
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

def process_prompts(model, tokenizer, prompts, max_new_tokens, device, batch_size):
    """Process prompts through the model and return outputs"""
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing prompts"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated outputs
        for j, output_id in enumerate(output_ids):
            prompt_tokens_length = inputs.input_ids[j].shape[0]
            response_ids = output_id[prompt_tokens_length:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            results.append({
                "prompt": batch_prompts[j],
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
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.device)
    
    # Read prompts from input CSV
    prompts = read_prompts(args.input_csv, args.prompt_column)
    print(f"Loaded {len(prompts)} prompts from {args.input_csv}")
    
    # Process prompts through the model
    results = process_prompts(model, tokenizer, prompts, args.max_new_tokens, args.device, args.batch_size)
    
    # Save results to output CSV
    save_results(results, args.output_csv)

if __name__ == "__main__":
    main()