import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from nnsight import LanguageModel
from tqdm import tqdm

# Example usage
# CUDA_VISIBLE_DEVICES=0 python -m probing.activations --layers 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31 --dataset_path dataset/non_cautious.csv --output_dir activations/prompt/ --type prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Extract residual stream activations from DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument('--layers', type=str, default='15,19,23,27,31',
                        help='Comma-separated list of layer numbers to extract activations from')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing tokens')
    parser.add_argument('--dataset_path', type=str, default='dataset/non_cautious.csv',
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='activations/',
                        help='Directory to save the activations')
    parser.add_argument('--max_tokens', type=int, default=150,
                        help='Maximum number of tokens to process per example')
    parser.add_argument('--type', type=str, default='cot', help="CoT tokens (cot) or 3 tokens at the end of prompt (baseline) or whole prompt (prompt)")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()
    # Parse layers
    layers = [int(layer) for layer in args.layers.split(',')]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the CSV dataset directly
    print(f"Loading CSV dataset from {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)

    print(f"Processing {len(df)} examples")

    # Initialize model
    model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    print(f"Initializing model {model_name}")
    model = LanguageModel(model_name, device_map="auto")

    # Initialize dictionary to store activation matrices for each layer
    activation_matrices = {layer: [] for layer in layers}
    print(f"Caching activations mode: {args.type}")

    # Process each example
    for idx, row in enumerate(tqdm(df.itertuples())):
        chat = [{"role": "user", "content": row.forbidden_prompt}]
        prompt_tokens = model.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        
        if args.type == 'cot':
            # Encode the response separately
            response_tokens = model.tokenizer.encode(row.response, add_special_tokens=False)
            # We want the first 150 tokens of the CoT (response)
            tokens_to_process = prompt_tokens + response_tokens[:args.max_tokens]
            target_start = len(prompt_tokens)  # Start of CoT
            target_end = len(tokens_to_process)  # End of our selection
        elif args.type == 'baseline':
            tokens_to_process = prompt_tokens
            target_start = max(0, len(prompt_tokens) - 3)  # Last 3 tokens of prompt
            target_end = len(prompt_tokens)
        elif args.type == 'prompt':
            tokens_to_process = prompt_tokens
            target_start = 0
            target_end = len(prompt_tokens)
        else:
            print("WARNING args.type not selected. Your choices are cot, baseline, prompt.")


        # Process the entire sequence at once to avoid batch alignment issues
        input_text = model.tokenizer.decode(tokens_to_process)
        
        # Initialize dict to collect activations for this example across all layers
        example_layer_activations = {layer: [] for layer in layers}
        
        with torch.no_grad():
            with model.trace(input_text):
                for layer in layers:
                    activation = model.model.layers[layer].input_layernorm.input.save()
                    example_layer_activations[layer].append(activation)
        
        # Compute means and add to matrices
        for layer in layers:
            layer_activations = example_layer_activations[layer][0]
            select_tokens = layer_activations[:, target_start:target_end, :]
            
            print(f"Selected tokens shape: {select_tokens.shape}")
            
            # Compute mean across tokens (dimension 1)
            mean_activation = torch.mean(select_tokens, dim=1).detach().cpu().numpy()
            activation_matrices[layer].append(mean_activation.squeeze())
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Save activation matrices for each layer
    for layer, activations in activation_matrices.items():
        if activations:
            activation_matrix = np.stack(activations)
            output_path = os.path.join(args.output_dir, f"deepseek_layer_{layer}_noncautious_activations.npy")
            np.save(output_path, activation_matrix)
            
            print(f"Saved activation matrix for layer {layer} with shape {activation_matrix.shape} to {output_path}")
    
    print("Extraction complete.")

def run():
    main()

if __name__ == "__main__":
    main()