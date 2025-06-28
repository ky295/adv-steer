import argparse
import gc

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.ortho_csv_generation --model_name 'kureha295/ortho_model' --input_csv 'dataset/cautious_eval.csv' --output_csv 'dataset/orthogonalized_outputs_2048.csv' --max_new_tokens 2048

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through the orthogonalised model"
    )
    parser.add_argument("--model_name", type=str, default='kureha295/ortho_model', help="Load the model")
    parser.add_argument('--input_csv', type=str, default='dataset/cautious_eval.csv',    
                        help='Path to the input CSV file with prompts')
    parser.add_argument('--output_csv', type=str, default='dataset/orthogonalized_outputs_2048.csv',
                        help='Path to save the output CSV file')
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    return parser.parse_args()

def read_csv(input_csv):
    # Read prompts from the CSV file
    print(f"Reading prompts from {input_csv}...")
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(input_csv)
        prompts = df['forbidden_prompt'].tolist()
        print(f"Loaded {len(prompts)} prompts from the CSV file")
        return prompts
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return None

def save_csv(results, output_csv):
    # Save results to CSV
    print(f"\nSaving results to {output_csv}...")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        print(f"Results saved successfully to {output_csv}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
    
    return None

def load_model(model_name, device):
    """Load the model and tokenizer"""
    print("Loading model and tokenizer...")
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
    raw_tokenized_chat = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    op_length = len(raw_tokenized_chat)
    tokenized_chat = torch.tensor([raw_tokenized_chat]).to(device)
    return tokenized_chat, op_length

def gen_text(model, tokenizer, op_length, tokenized_chat, max_new_tokens):
    with torch.no_grad():
        output_ids = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
            do_sample=True, 
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id
        )
            
        # Decode the generated output
        response = tokenizer.decode(output_ids[0][op_length:], skip_special_tokens=True)
    return response

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

    prompts = read_csv(args.input_csv)
    results = []

    model, tokenizer = load_model(args.model_name, args.device)

    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}")
            
        # Apply chat template
        tokenized_chat, op_length = apply_chat_template(prompt, tokenizer, args.device)
            
        # Generate text with orthogonalized model
        print("Generating orthogonalized output...")
        orthogonalized_text = gen_text(model, tokenizer, op_length, tokenized_chat, args.max_new_tokens)
            
        # Save result
        results.append({"forbidden_prompt": prompt, "response": orthogonalized_text})
        del prompt, orthogonalized_text
        gc.collect()
        torch.cuda.empty_cache()

    save_csv(results, args.output_csv)

def run():
    main()

if __name__ == "__main__":
    main()