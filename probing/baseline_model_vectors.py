"""
This script visualizes the cosine similarity between a pre-computed direction vector
and per-token activations from layer 18 of DeepSeek-R1-Distill-Llama-8B.
"""
# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.baseline_model_vectors
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import gc
from nnsight import LanguageModel
from matplotlib.ticker import MaxNLocator
import csv
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize activation similarities")
    parser.add_argument("--vector_path", type=str, default="probing/cautious_dir.pt", 
                        help="Path to the pre-computed direction vector")
    parser.add_argument("--ortho", type=bool, default=False, help="Are you using a local model? (y/n)")
    parser.add_argument("--layer", type=int, default=18, 
                        help="Layer to extract activations from")
    parser.add_argument("--output_dir", type=str, default="figures/non_cautious_heatmap",
                        help="Directory for output files")
    
    return parser.parse_args()

def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

def get_activations(model, input_text, layer=18):
    """Extract activations from specified layer using NNsight."""
    # Clear memory before processing
    gc.collect()
    torch.cuda.empty_cache()
    
    # Tokenize input text
    tokens = model.tokenizer.encode(input_text)
    token_texts = [model.tokenizer.decode([token]) for token in tokens]
    # Trace the model to get activations
    with model.trace(input_text) as tracer:
        # Save activations from the residual stream
        activation_proxy = model.model.layers[layer].input_layernorm.input.save()
    
    # Ensure proper tensor format
    activation_tensor = activation_proxy
    
    # Print detailed tensor information
    print(f"Activation tensor type: {type(activation_tensor)}")
    print(f"Activation tensor shape: {activation_tensor.shape}")
    
    # If tensor is flattened, try to reshape it
    if len(activation_tensor.shape) == 1:
        # Calculate the expected sequence length
        seq_len = len(tokens)
        hidden_size = 4096  # Expected hidden size for DeepSeek-R1-Distill-Llama-8B
        
        print(f"Tensor appears to be flattened. Attempting reshape...")
        print(f"Expected shape: [{seq_len}, {hidden_size}]")
        
        # Check if reshaping is possible
        if activation_tensor.numel() == seq_len * hidden_size:
            activation_tensor = activation_tensor.reshape(seq_len, hidden_size)
            print(f"Reshaped tensor to: {activation_tensor.shape}")
        else:
            print(f"WARNING: Cannot reshape tensor to expected dimensions.")
            print(f"Tensor has {activation_tensor.numel()} elements, but expected {seq_len * hidden_size}")
    
    # Clear cache after processing
    gc.collect()
    torch.cuda.empty_cache()
    
    return activation_tensor, token_texts

def compute_cosine_similarities(activation_tensor, direction_vector):
    """Compute cosine similarity between direction vector and each token's activation."""
    # Clear memory before computation
    gc.collect()
    torch.cuda.empty_cache()
    
    # Remove batch dimension if present
    if len(activation_tensor.shape) == 3:
        activation_tensor = activation_tensor.squeeze(0)  # Now shape is [463, 4096]
    
    # Ensure direction vector has the right shape
    direction_vector = direction_vector.reshape(1, -1)  # Shape [1, 4096]
    
    # Compute cosine similarity for each token
    similarities = []
    for token_idx in range(activation_tensor.shape[0]):
        token_activation = activation_tensor[token_idx].reshape(1, -1)  # Shape [1, 4096]
        similarity = torch.nn.functional.cosine_similarity(
            token_activation, direction_vector, dim=1
        ).item()
        similarities.append(similarity)
    
    # Clear memory after computation
    gc.collect()
    torch.cuda.empty_cache()
    
    return similarities

def plot_heatmap(token_texts, similarities, output_path="similarity_heatmap.png"):
    """Create a heatmap visualization of token similarities."""
    plt.figure(figsize=(12, 4))
    
    # Reshape similarities for heatmap (as a row)
    sim_matrix = np.array(similarities).reshape(1, -1)
    
    # Create heatmap
    ax = sns.heatmap(sim_matrix, cmap='coolwarm', center=0, 
                   xticklabels=token_texts, yticklabels=["Similarity"], vmin=-0.4, vmax=0.4)
    
    # Rotate x-axis labels
    plt.xticks(rotation=90, fontsize=8)
    
    plt.title('Similarity Heatmap: Direction Vector vs. Token Activations', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")
    plt.close()
    
    # Clear memory after plotting
    gc.collect()

def read_csv(input_csv):
    # Read prompts from the CSV file
    print(f"Reading responses from {input_csv}...")
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(input_csv)
        prompts = df['response'].tolist()
        print(f"Loaded {len(prompts)} prompts from the CSV file")
        return prompts
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return None

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initial memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Set plotting settings
    set_plotting_settings()
    
    # Load the pre-computed direction vector
    direction_vector = torch.load(args.vector_path)
    print(f"Loaded direction vector with shape: {direction_vector.shape}")
    if args.ortho == True: 
        model_name = "kureha295/ortho_model"
        print(f"Initializing model {model_name}")
    else:
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        print(f"Initializing model {model_name}")

    model = LanguageModel(model_name, device_map="auto")
    input_csv = "dataset/non_cautious.csv"
    
    # Memory cleanup after model loading
    gc.collect()
    torch.cuda.empty_cache()

    prompts = read_csv(input_csv)[:10]
    
    for i, text in enumerate(prompts):
        # Get activations for the input text
        print(f"Extracting activations from layer {args.layer}")
        text = text[:1000]
        activations, token_texts = get_activations(model, text, layer=args.layer)
        
        # Compute cosine similarities
        similarities = compute_cosine_similarities(activations, direction_vector)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create visualizations
        plot_heatmap(token_texts, similarities, 
                    output_path=os.path.join(args.output_dir, f"non_cautious_{i}_heatmap.png"))
        
        # Print highest and lowest similarity tokens
        sorted_indices = np.argsort(similarities)
        print("\nTokens with highest similarity:")
        for idx in sorted_indices[-5:]:
            print(f"  {token_texts[idx]}: {similarities[idx]:.4f}")
        
        print("\nTokens with lowest similarity:")
        for idx in sorted_indices[:5]:
            print(f"  {token_texts[idx]}: {similarities[idx]:.4f}")
        
        # Final memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

def run():
    main()

if __name__ == "__main__":
    main()