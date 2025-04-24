import torch
import einops
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple
import argparse
import numpy as np
import gc

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.fix_optimised_ortho

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through DeepSeek-R1-Distill-Llama-8B model"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="activations/",
        help="Path to activations directory" 
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Load the model")
    parser.add_argument("--layer", type=int, default=18, help="Layer to take the activations")
    parser.add_argument('--input_csv', type=str, default='dataset/cautious.csv',    
                        help='Path to the input CSV file with prompts')
    parser.add_argument('--output_csv', type=str, default='dataset/orthogonalized_outputs.csv',
                        help='Path to save the output CSV file')
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    return parser.parse_args()

def get_orthogonalized_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalize a matrix with respect to a vector direction.
    
    This removes the component of each row in the matrix that projects onto vec,
    making the matrix unable to produce outputs in the direction of vec.
    
    Args:
        matrix: The matrix to orthogonalize (any shape where the last dimension is d_model)
        vec: The unit vector to orthogonalize against (shape [d_model])
        
    Returns:
        The orthogonalized matrix with the same shape as the input matrix
    """
    
    # For DeepSeek model compatibility, we need to handle different matrix orientations
    # For embedding and MLP Down Projection, the input is [vocab/hidden, d_model]
    # For attention output projection, the input is [d_model, d_model]
    
    # Compute the projection of each row of the matrix onto vec
    # Shape: [..., 1]
    projection = torch.matmul(matrix, vec.unsqueeze(-1))
    
    # Multiply by vec to get the component in the direction of vec
    # Shape: [..., d_model]
    component = projection * vec.view((1,) * (matrix.dim() - 1) + (-1,))
    
    # Subtract the component to get the orthogonalized matrix
    return matrix - component

def apply_orthogonalization(model: AutoModelForCausalLM, direction_vector: torch.Tensor) -> AutoModelForCausalLM:
    """
    Apply orthogonalization to all weight matrices that write to the residual stream.
    
    Args:
        model: The HuggingFace model to modify
        direction_vector: The direction vector to orthogonalize against
        
    Returns:
        The modified model
    """
    # 1. Orthogonalize embedding weights
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction_vector
    )
    
    # 2. Orthogonalize all attention output projection and MLP output projection weights
    for layer in model.model.layers:
        # Attention output projection (o_proj)
        layer.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            layer.self_attn.o_proj.weight.data, direction_vector
        )
        
        # MLP output projection (down_proj)
        layer.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            layer.mlp.down_proj.weight.data, direction_vector
        )
    
    return model

def main():
    # Load the direction vector
    direction_path = "probing/cautious_dir.pt"
    if not os.path.exists(direction_path):
        raise FileNotFoundError(f"Direction vector file not found at {direction_path}")
    
    direction_vector = torch.load(direction_path, map_location="cpu")
    print(f"Loaded direction vector with shape: {direction_vector.shape}")
    
    # Load the model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Using float16 for memory efficiency
        device_map="auto"  # Let the library decide optimal device placement
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move direction vector to the same device as the model's first parameter
    first_param = next(model.parameters())
    direction_vector = direction_vector.to(first_param.device).to(first_param.dtype)
    
    # Apply orthogonalization
    print("Applying orthogonalization to model weights...")
    model = apply_orthogonalization(model, direction_vector)
    print("Orthogonalization complete.")
    
    # Save the modified model
    output_dir = "orthogonalized_model"
    print(f"Saving orthogonalized model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

def run():
    main()

if __name__ == "__main__":
    main()