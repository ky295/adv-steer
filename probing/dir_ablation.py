import argparse
import functools
import gc
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.dir_ablation --layer 17


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts with directional ablation in DeepSeek model"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="activations/cot150/",
        help="Path to activations directory" 
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Load the model")
    parser.add_argument("--layer", type=int, default=17, help="Layer to extract direction from")
    parser.add_argument('--input_csv', type=str, default='dataset/cautious_eval.csv',    
                        help='Path to the input CSV file with prompts')
    parser.add_argument('--output_csv', type=str, default='dataset/directional_ablation_outputs.csv',
                        help='Path to save the output CSV file')
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
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

def read_csv(input_csv):
    # Read prompts from the CSV file
    print(f"Reading prompts from {input_csv}...")
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(input_csv)
        # For debugging purposes, only process the first 5 examples
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

def apply_chat_template(prompt, tokenizer, device):
    chat = [{"role": "user", "content": prompt}]
    raw_tokenized_chat = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    op_length = len(raw_tokenized_chat)
    tokenized_chat = torch.tensor([raw_tokenized_chat]).to(device)
    return tokenized_chat, op_length

def load_activations(file_path):
    """Load activation data from .npy file"""
    try:
        activations = np.load(file_path)
        print(f"Loaded activations with shape: {activations.shape}")
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
    """Get normalized direction vector for ablation"""
    dir = mean_act1 - mean_act2
    # Normalize the direction for ablation
    dir = dir / dir.norm()
    return dir

# Directional ablation hook function
def directional_ablation_hook(
    module: torch.nn.Module,
    input: List[torch.Tensor],
    output,
    direction: torch.Tensor
):
    """
    Hook function that applies directional ablation to remove a specific direction.
    Implements: x' = x - (c_hat @ c_hat^T) @ x 
    
    Args:
        module: The module whose forward pass is being hooked
        input: Inputs to the module
        output: Output of the module (can be tuple or tensor)
        direction: The normalized direction to ablate (unit vector)
    
    Returns:
        Modified output with the direction ablated
    """
    hidden_states = output[0]
    other_outputs = output[1:] if len(output) > 1 else ()
    
    # Get original shape
    original_shape = hidden_states.shape
    
    # Reshape to 2D for easier computation: [total_tokens, hidden_dim]
    batch_size, seq_len, hidden_dim = original_shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
    
    # Apply directional ablation: x' = x - c_hat * (c_hat^T @ x)
    # Compute projection: c_hat^T @ x for each token
    projections = torch.matmul(hidden_states_flat, direction)  # [total_tokens]
    
    # Compute ablated activations: x - c_hat * projection
    ablated_flat = hidden_states_flat - torch.outer(projections, direction)
    
    # Reshape back to original shape
    ablated_states = ablated_flat.view(batch_size, seq_len, hidden_dim)

    return (ablated_states,) + other_outputs

def register_directional_ablation_hooks(model, direction):
    """
    Register hooks for directional ablation on ALL transformer layers.
    
    Args:
        model: The model to add hooks to
        direction: The normalized direction to ablate
    
    Returns:
        handles: List of hook handles for removal later
    """
    handles = []
    hook_fn = functools.partial(directional_ablation_hook, direction=direction)
    
    # Hook ALL transformer layers
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(hook_fn))
    
    print(f"Registered directional ablation hooks on {len(handles)} layers")
    print(f"Direction norm: {direction.norm().item():.6f} (should be ~1.0 for normalized)")
    return handles

def gen_text(model, tokenizer, op_length, tokenized_chat, max_new_tokens, hooks=None):
    """Generate text with optional hooks for directional ablation"""
    
    # If hooks are provided, they're already registered and no need to do anything here
    with torch.no_grad():
        output_ids = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
            do_sample = True,
            temperature = 0.6,
            pad_token_id=tokenizer.eos_token_id
        )
            
        # Decode the generated output
        response = tokenizer.decode(output_ids[0][op_length:], skip_special_tokens=True)
    
    return response

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Using directional ablation across ALL layers")
    gc.collect()
    torch.cuda.empty_cache()

    # Load activations
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
    
    # Calculate normalized cautious direction for ablation
    cautious_dir = get_dir(cautious_mean_act, noncautious_mean_act)
    print(f"Cautious direction shape: {cautious_dir.shape}")
    print(f"Cautious direction norm: {torch.norm(cautious_dir).item():.6f} (normalized)")
    
    # Free memory before generating ablated text
    del cautious_mean_act, noncautious_mean_act
    gc.collect()
    torch.cuda.empty_cache()

    # # Alternative: Load tensor directly from cautious_dir.pt and normalize
    # cautious_dir = torch.load('probing/cautious_dir.pt')
    # cautious_dir = cautious_dir / cautious_dir.norm()  # Normalize for ablation
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name, args.device)

    # Register hooks for directional ablation on ALL layers
    ablation_hooks = register_directional_ablation_hooks(model, cautious_dir)

    prompts = read_csv(args.input_csv)
    results = []

    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}")
            
        # Apply chat template
        tokenized_chat, op_length = apply_chat_template(prompt, tokenizer, args.device)
            
        # Generate text with directional ablation
        print("Generating text with directional ablation...")
        ablated_text = gen_text(model, tokenizer, op_length, tokenized_chat, args.max_new_tokens)
            
        # Save result
        results.append({
            "forbidden_prompt": prompt, 
            "response": ablated_text,
        })
        del prompt, ablated_text
        gc.collect()
        torch.cuda.empty_cache()

    save_csv(results, args.output_csv)
    
    # Remove hooks to clean up
    for handle in ablation_hooks:
        handle.remove()
    
    print("Directional ablation completed successfully!")

def run():
    main()

if __name__ == "__main__":
    main()