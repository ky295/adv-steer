import torch
import pickle
import numpy as np
from tqdm import tqdm
import os
from nnsight import LanguageModel
import argparse

# Example usage
# CUDA_VISIBLE_DEVICES=0 python -m probing.activations --layers 15,19,23,27,31 --dataset_path dataset/tokenized_deepseek_dataset.pkl --output_dir activations/

def parse_args():
    parser = argparse.ArgumentParser(description="Extract residual stream activations from DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument('--layers', type=str, default='15,19,23,27,31',
                        help='Comma-separated list of layer numbers to extract activations from')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing tokens')
    parser.add_argument('--dataset_path', type=str, default='../dataset/tokenized_deepseek_dataset.pkl',
                        help='Path to the tokenized dataset')
    parser.add_argument('--output_dir', type=str, default='../activations/',
                        help='Directory to save the activations')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='Maximum number of tokens to process per example')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples to process (None for all)')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Parse layers
    layers = [int(l) for l in args.layers.split(',')]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the tokenized dataset
    print(f"Loading tokenized dataset from {args.dataset_path}")
    with open(args.dataset_path, 'rb') as f:
        tokenized_dataset = pickle.load(f)
    
    # Limit number of examples if specified
    if args.max_examples is not None:
        tokenized_dataset = tokenized_dataset[:args.max_examples]
    
    print(f"Processing {len(tokenized_dataset)} examples")
    
    # Initialize model
    model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    print(f"Initializing model {model_name}")
    model = LanguageModel(model_name, device_map="auto")
    
    # Initialize dictionary to store activation matrices for each layer
    activation_matrices = {layer: [] for layer in layers}
    
    # Process each example
    for idx, tokens in enumerate(tqdm(tokenized_dataset)):
        # Limit token length
        tokens = tokens[:args.max_tokens]
        
        # Initialize dict to collect activations for this example across all layers
        example_layer_activations = {layer: [] for layer in layers}
        
        # Process in batches to avoid memory issues
        for start_idx in range(0, len(tokens), args.batch_size):
            end_idx = min(start_idx + args.batch_size, len(tokens))
            batch_tokens = tokens[start_idx:end_idx]
            
            # Convert tokens to string for model input
            # DeepSeek model expects text input, not token IDs directly
            try:
                # First try: If tokenized_dataset contains token IDs, decode them
                if hasattr(model.tokenizer, 'decode'):
                    input_text = model.tokenizer.decode(batch_tokens)
                else:
                    # Fallback if we can't decode
                    input_text = " ".join(map(str, batch_tokens))
            except:
                # In case of errors, use raw tokens as string
                input_text = " ".join(map(str, batch_tokens))
            
            # Run forward pass with NNsight
            with model.trace(input_text) as tracer:
                # Save activations for each layer
                for layer in layers:
                    # DeepSeek-R1-Distill-Llama-8B uses input_layernorm.input for residual stream
                    # The structure is model.layers[layer_num].input_layernorm.input
                    activation = model.model.layers[layer].input_layernorm.input.save()
                    example_layer_activations[layer].append(activation)
        
        # Compute means and add to matrices
        for layer in layers:
            if example_layer_activations[layer]:
                # Concatenate batches
                try:
                    # For newer versions of PyTorch/NNsight that return tensors directly
                    layer_activations = torch.cat(example_layer_activations[layer], dim=1)
                except:
                    # Fallback for versions that might wrap tensors in tuples/lists
                    layer_activations = torch.cat([act[0] if isinstance(act, (list, tuple)) else act 
                                                  for act in example_layer_activations[layer]], dim=1)
                
                # Compute mean across tokens (dimension 1)
                mean_activation = torch.mean(layer_activations, dim=1).detach().cpu().numpy()
                activation_matrices[layer].append(mean_activation.squeeze())
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(tokenized_dataset)} examples")
            
            # Print some stats about the first layer as a sanity check
            if layers[0] in activation_matrices and activation_matrices[layers[0]]:
                first_layer_shape = activation_matrices[layers[0]][0].shape
                print(f"First layer activation shape for first example: {first_layer_shape}")
                print(f"First layer activation mean: {np.mean(activation_matrices[layers[0]][0]):.6f}")
    
    # Save activation matrices for each layer
    for layer, activations in activation_matrices.items():
        if activations:
            activation_matrix = np.stack(activations)
            output_path = os.path.join(args.output_dir, f"deepseek_layer_{layer}_activations.npy")
            np.save(output_path, activation_matrix)
            
            print(f"Saved activation matrix for layer {layer} with shape {activation_matrix.shape} to {output_path}")
    
    # Save metadata
    metadata = {
        "model": model_name,
        "layers": layers,
        "num_examples": len(tokenized_dataset),
        "max_tokens": args.max_tokens,
        "shapes": {layer: activation_matrices[layer][0].shape if activation_matrices[layer] else None
                  for layer in layers}
    }
    
    with open(os.path.join(args.output_dir, "deepseek_extraction_metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Extraction complete.")

def run():
    main()

if __name__ == "__main__":
    main()