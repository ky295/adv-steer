import torch
import pickle
import numpy as np
from tqdm import tqdm
import os
from nnsight import LanguageModel
import argparse
import pandas as pd
import gc

# Example usage
# CUDA_VISIBLE_DEVICES=0 python -m probing.activations --layers 3,7,11,15,17,19,23,27,31 --dataset_path dataset/non_cautious.csv --output_dir activations/cot150/ --cot

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
    parser.add_argument('--cot', type = bool, help="CoT tokens or 3 tokens at the end of prompt")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()
    # Parse layers
    layers = [int(l) for l in args.layers.split(',')]
    
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

    # Process each example
    for idx, row in enumerate(tqdm(df.itertuples())):
        chat = [{"role": "user", "content": row.forbidden_prompt}]
        # tokens = model.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        prompt_tokens = model.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        # Mark where CoT begins (exact boundary)
        cot_start_idx = len(prompt_tokens)
        if args.cot:
            # Encode the response separately
            response_tokens = model.tokenizer.encode(row.response, add_special_tokens=False)
            # Combine tokens (this preserves the exact boundary)
            tokens = prompt_tokens + response_tokens
            end_idx= cot_start_idx + args.max_tokens
            # Mark where CoT begins (exact boundary)
            start_pos = cot_start_idx
        else:
            tokens = prompt_tokens
            end_idx = cot_start_idx
            start_pos = cot_start_idx - 3

        # Initialize dict to collect activations for this example across all layers
        example_layer_activations = {layer: [] for layer in layers}
        token_positions = []

        # Process in batches to avoid memory issues
        for start_idx in range(0, end_idx, args.batch_size):
            batch_idx = min(start_idx + args.batch_size, end_idx)
            batch_tokens = tokens[start_idx:batch_idx]
            token_positions.append((start_idx, batch_idx))
            # Convert batch tokens to input text
            input_text = model.tokenizer.decode(batch_tokens)
            # Run forward pass with NNsight
            with torch.no_grad():  # Disable gradient tracking to save memory
                with model.trace(input_text) as tracer:
                    # Save activations for each layer
                    for layer in layers:
                        # DeepSeek-R1-Distill-Llama-8B uses input_layernorm.input for residual stream
                        # The structure is model.layers[layer_num].input_layernorm.input
                        activation = model.model.layers[layer].input_layernorm.input.save()
                        example_layer_activations[layer].append(activation)

            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()
        
        # Compute means and add to matrices
        for layer in layers:
            layer_activations = torch.cat(example_layer_activations[layer], dim=1)
            # Get the last 3 tokens of the sequence OR args.max_tokens number of tokens in the CoT
            select_tokens = layer_activations[:, start_pos: end_idx, :]
            # Compute mean across tokens (dimension 1)
            mean_activation = torch.mean(select_tokens, dim=1).detach().cpu().numpy()

            activation_matrices[layer].append(mean_activation.squeeze())
        
        # Clear CUDA cache after each batch
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