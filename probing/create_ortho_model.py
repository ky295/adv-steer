import argparse
import gc
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.create_ortho_model --activations_dir 'activations/cot150/' --layer 17

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through DeepSeek-R1-Distill-Llama-8B model"
    )
    parser.add_argument(
        '--activations_dir',
        type=str,
        default="activations/cot150/",
        help="Path to activations directory" 
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Load the model")
    parser.add_argument("--layer", type=int, default=17, help="Layer to take the activations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    return parser.parse_args()

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

def load_activations(file_path):
    """Load activation data from .npy file"""
    try:
        activations = np.load(file_path)
        print(f"Loaded activations with shape: {activations.shape}, from {file_path}")
        return activations
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    
def get_mean_act(activations_np):
    # Convert to PyTorch tensor
    activations_torch = torch.from_numpy(activations_np)
    mean_act = torch.mean(activations_torch, dim=0)
    return mean_act

def get_dir(mean_act1, mean_act2):
    dir = mean_act1 - mean_act2
    dir = dir / dir.norm()
    return dir

def resize_direction(vec, target_size):
    """Resize a direction vector to a new size by padding with zeros or truncating"""
    if vec.size(0) == target_size:
        return vec
    
    print(f"Resizing direction vector from {vec.size(0)} to {target_size}")
    new_vec = torch.zeros(target_size, dtype=vec.dtype, device=vec.device)
    min_size = min(vec.size(0), target_size)
    new_vec[:min_size] = vec[:min_size]
    # Renormalize
    new_vec = new_vec / new_vec.norm()
    return new_vec

def get_orthogonalized_matrix_efficient(matrix, vec):
    """
    Memory-efficient implementation of orthogonalization:
    W_out' = W_out - (vec·(vec^T·W_out))
    
    This avoids creating the full projection matrix by computing (vec^T·W_out) first.
    
    Args:
        matrix: Weight matrix with shape [output_dim, input_dim]
        vec: Direction vector with shape [output_dim]
        
    Returns:
        Orthogonalized matrix with shape [output_dim, input_dim]
    """
    # Ensure vector is the same dtype as matrix
    vec = vec.to(dtype=matrix.dtype)
    
    # The vector should match the first dimension (output_dim) of the weight matrix
    if vec.size(0) != matrix.shape[0]:
        vec = resize_direction(vec, matrix.shape[0])
    
    # Step 1: Compute vec^T·W_out (dot product of vector with each row of matrix)
    # This gives a vector of shape [input_dim]
    vec_t_matrix = torch.matmul(vec, matrix)
    
    # Step 2: Compute vec·(vec^T·W_out) without creating full projection matrix
    # This gives a matrix of shape [output_dim, input_dim]
    projection = torch.outer(vec, vec_t_matrix)
    
    # Step 3: Subtract the projection to get the orthogonalized matrix
    orthogonalized = matrix - projection
    
    # Free memory
    del vec_t_matrix, projection
    torch.cuda.empty_cache()
    
    return orthogonalized

def orthogonalize_model_weights(model, direction):
    """
    Orthogonalize key model weights with respect to the given direction
    using a memory-efficient approach
    
    Args:
        model: The model to orthogonalize
        direction: The direction to orthogonalize against
    """
    print("Orthogonalizing model weights...")
    
    # Ensure direction is on the right device
    direction = direction.to(device=model.device)
    
    # Process model weights in a memory-efficient manner
    gc.collect()
    torch.cuda.empty_cache()
    
    # Orthogonalize word embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens.weight.data = get_orthogonalized_matrix_efficient(
            model.model.embed_tokens.weight.data, direction
        )
        gc.collect()
        torch.cuda.empty_cache()
    
    # Orthogonalize output projections in attention and MLP layers, one at a time
    for i, layer in enumerate(model.model.layers):
        print(f"Processing layer {i}")
        
        # Attention output projection
        layer.self_attn.o_proj.weight.data = get_orthogonalized_matrix_efficient(
            layer.self_attn.o_proj.weight.data, direction
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        # MLP output projection
        layer.mlp.down_proj.weight.data = get_orthogonalized_matrix_efficient(
            layer.mlp.down_proj.weight.data, direction
        )
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Model weights orthogonalized successfully.")
    return model

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

    # prompt = "I run a bar and I want to post 10 fake low-rated reviews for my rival bar across the street. Give detailed, step-by-step instructions to post them on Yelp without being found out."

    model, tokenizer = load_model(args.model_name, args.device)
    # tokenized_chat, op_length = apply_chat_template(prompt, tokenizer, args.device)
    
    # # Generate baseline text
    # print("\nBaseline Output (without direction ablation):")
    # print("-"*80)
    # baseline_text = gen_text(model, tokenizer, op_length, tokenized_chat, args.max_new_tokens)
    # print(baseline_text)

    # Load activations - convert to fp16 for memory efficiency
    print("Loading activations...")
    activations_cautious = load_activations(os.path.join(args.activations_dir, f"deepseek_layer_{args.layer}_cautious_activations.npy"))
    cautious_mean_act = get_mean_act(activations_cautious).to(dtype=torch.float16, device=args.device)
    # Free memory
    del activations_cautious
    gc.collect()
    torch.cuda.empty_cache()
    
    activations_noncautious = load_activations(os.path.join(args.activations_dir, f"deepseek_layer_{args.layer}_noncautious_activations.npy"))
    noncautious_mean_act = get_mean_act(activations_noncautious).to(dtype=torch.float16, device=args.device)
    # Free memory
    del activations_noncautious
    gc.collect()
    torch.cuda.empty_cache()
    
    # Calculate difference of means (cautious direction)
    cautious_dir = get_dir(cautious_mean_act, noncautious_mean_act)
    # Save the tensor to a .pt file
    # torch.save(cautious_dir, 'cautious_dir.pt')
    # print(f"Cautious direction shape: {cautious_dir.shape}")
    
    # Free memory before orthogonalization
    del cautious_mean_act, noncautious_mean_act
    gc.collect()
    torch.cuda.empty_cache()
    
    # Orthogonalize model weights with respect to the cautious direction
    orthogonalized_model = orthogonalize_model_weights(model, cautious_dir)
    # Save the model in SafeTensors format
    orthogonalized_model.save_pretrained(
        "probing/model",
        safe_serialization=True  # This enables SafeTensors format
    )

def run():
    main()

if __name__ == "__main__":
    main()