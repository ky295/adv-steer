import os
import argparse
import numpy as np
import torch
import einops
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Callable, Dict, Any
import gc
import csv
import pandas as pd

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.activation_addition


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
    parser.add_argument("--layer", type=int, default=18, help="Layer to take the activations")
    parser.add_argument('--input_csv', type=str, default='dataset/non_cautious.csv',    
                        help='Path to the input CSV file with prompts')
    parser.add_argument('--output_csv', type=str, default='dataset/ablated_outputs.csv',
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
        df = df.head(5)
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
    dir = mean_act1 - mean_act2
    return dir

# Directional ablation hook function
def direction_ablation_hook(
    module: torch.nn.Module,
    input: List[torch.Tensor],
    output: torch.Tensor,
    direction: torch.Tensor
) -> torch.Tensor:
    """
    Hook function that ablates a specific direction from activations.
    
    Args:
        module: The module whose forward pass is being hooked
        input: Inputs to the module
        output: Output of the module
        direction: The direction to ablate (must be a unit vector)
    
    Returns:
        Modified output with the direction ablated
    """
    # Calculate the projection onto the direction
    # Ensure both tensors are on the same device
    direction = direction.to(output.device)
    
    # Reshape output for easier computation if needed
    original_shape = output.shape
    if len(original_shape) > 2:
        # Reshape to [batch_size * seq_len, hidden_dim]
        output_reshaped = output.view(-1, output.size(-1))
    else:
        output_reshaped = output
    
    # Add the projection from the output
    output_ablated = output_reshaped + (0.5 * direction)
    
    # Reshape back to original shape if needed
    if len(original_shape) > 2:
        output_ablated = output_ablated.view(original_shape)
    
    return output_ablated

def register_ablation_hooks(model, direction, layer_names=None):
    """
    Register hooks for directional ablation on model layers.
    
    Args:
        model: The model to add hooks to
        direction: The direction to ablate
        layer_names: Optional list of specific layer names to hook 
                    (default: hooks all self-attention outputs and MLP outputs)
    
    Returns:
        handles: List of hook handles for removal later
    """
    handles = []
    hook_fn = functools.partial(direction_ablation_hook, direction=direction)
    
    # Default behavior: hook all self-attention output projections and MLP output projections
    if layer_names is None:
        # For each transformer layer
        for i, layer in enumerate(model.model.layers):
            # Hook self-attention output projection
            handles.append(layer.self_attn.o_proj.register_forward_hook(hook_fn))
            
            # Hook MLP output projection
            handles.append(layer.mlp.down_proj.register_forward_hook(hook_fn))
    else:
        # Hook specific named modules
        for name in layer_names:
            module = model.get_submodule(name)
            handles.append(module.register_forward_hook(hook_fn))
    
    print(f"Registered {len(handles)} ablation hooks")
    return handles

def gen_text(model, tokenizer, op_length, tokenized_chat, max_new_tokens, hooks=None):
    """Generate text with optional hooks for activation ablation"""
    
    # If hooks are provided, they're already registered and no need to do anything here
    with torch.no_grad():
        output_ids = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding
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
    
    # Calculate difference of means (cautious direction)
    cautious_dir = get_dir(cautious_mean_act, noncautious_mean_act)
    print(f"Cautious direction shape: {cautious_dir.shape}")
    
    # Free memory before generating ablated text
    del cautious_mean_act, noncautious_mean_act
    gc.collect()
    torch.cuda.empty_cache()

    # # Load tensor directly from cautious_dir.pt
    # cautious_dir = torch.load('probing/cautious_dir.pt')
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name, args.device)

    # Register hooks for direction ablation
    ablation_hooks = register_ablation_hooks(model, cautious_dir)

    prompts = read_csv(args.input_csv)
    results = []

    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}")
            
        # Apply chat template
        tokenized_chat, op_length = apply_chat_template(prompt, tokenizer, args.device)
            
        # Generate text with orthogonalized model
        print("Generating orthogonalized output...")
        ablated_text = gen_text(model, tokenizer, op_length, tokenized_chat, args.max_new_tokens)
            
        # Save result
        results.append({"forbidden_prompt": prompt, "response": ablated_text})
        del prompt, ablated_text
        gc.collect()
        torch.cuda.empty_cache()

    save_csv(results, args.output_csv)
    
    # Remove hooks to clean up
    for handle in ablation_hooks:
        handle.remove()

def run():
    main()

if __name__ == "__main__":
    main()