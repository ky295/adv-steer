import argparse
import functools
import gc
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.activation_addition --alpha 1.5 --layer 17


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts with activation addition in DeepSeek model"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="activations/cot150/",
        help="Path to activations directory" 
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Load the model")
    parser.add_argument("--layer", type=int, default=17, help="Layer to apply activation addition")
    parser.add_argument('--input_csv', type=str, default='dataset/non_cautious.csv',    
                        help='Path to the input CSV file with prompts')
    parser.add_argument('--output_csv', type=str, default='dataset/activation_addition_outputs.csv',
                        help='Path to save the output CSV file')
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Scaling factor for activation addition")
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
    dir = mean_act1 - mean_act2
    return dir

# Activation addition hook function
def activation_addition_hook(
    module: torch.nn.Module,
    input: List[torch.Tensor],
    output,
    direction: torch.Tensor,
    alpha: float = 1.0
):
    """
    Hook function that adds a direction to activations for steering.
    
    Args:
        module: The module whose forward pass is being hooked
        input: Inputs to the module
        output: Output of the module (can be tuple or tensor)
        direction: The direction to add (cautious direction vector)
        alpha: Scaling factor for the addition
    
    Returns:
        Modified output with the direction added
    """
    # Handle the case where output is a tuple (common for transformer layers)
    # The first element is typically the hidden states
    hidden_states = output[0]
    other_outputs = output[1:] if len(output) > 1 else ()
    
    # Ensure direction is on the same device as hidden_states
    direction = direction.to(hidden_states.device)
    
    # Get original shape for reshaping back later
    original_shape = hidden_states.shape
    # For transformer models, hidden_states shape is typically [batch_size, seq_len, hidden_dim]

    batch_size, seq_len, hidden_dim = original_shape
        
    # Add the direction to all token positions
    # Direction should be broadcast across batch and sequence dimensions
    direction_expanded = direction.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
    direction_expanded = direction_expanded.expand(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
    # Apply activation addition: x' = x + α * c
    hidden_states_modified = hidden_states + alpha * direction_expanded
        
    # Return the modified output in the same format as the input
    return (hidden_states_modified,) + other_outputs

def register_activation_addition_hooks(model, direction, target_layer, alpha=1.0):
    """
    Register hooks for activation addition on a specific layer.
    
    Args:
        model: The model to add hooks to
        direction: The direction to add (cautious direction)
        target_layer: The layer number to apply activation addition to
        alpha: Scaling factor for activation addition
    
    Returns:
        handles: List of hook handles for removal later
    """
    handles = []
    hook_fn = functools.partial(activation_addition_hook, direction=direction, alpha=alpha)
    
    # Hook the residual stream at the specified layer
    # This targets the output of the entire transformer block
    target_module = model.model.layers[target_layer]
    handles.append(target_module.register_forward_hook(hook_fn))
    
    print(f"Registered activation addition hook on layer {target_layer} with alpha={alpha}")
    return handles

def gen_text(model, tokenizer, op_length, tokenized_chat, max_new_tokens, hooks=None):
    """Generate text with optional hooks for activation addition"""
    
    # If hooks are provided, they're already registered and no need to do anything here
    with torch.no_grad():
        output_ids = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature = 0.6,
            pad_token_id=tokenizer.eos_token_id
        )
            
        # Decode the generated output
        response = tokenizer.decode(output_ids[0][op_length:], skip_special_tokens=True)
    
    return response

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using alpha={args.alpha} for activation addition")
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
    # This vector points from non-cautious to cautious
    cautious_dir = get_dir(cautious_mean_act, noncautious_mean_act)
    print(f"Cautious direction shape: {cautious_dir.shape}")
    print(f"Cautious direction norm: {torch.norm(cautious_dir).item():.4f}")
    
    # Free memory before generating text with activation addition
    del cautious_mean_act, noncautious_mean_act
    gc.collect()
    torch.cuda.empty_cache()

    # # Alternative: Load tensor directly from cautious_dir.pt
    # cautious_dir = torch.load('probing/cautious_dir.pt')
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name, args.device)

    # Register hooks for activation addition
    addition_hooks = register_activation_addition_hooks(
        model, cautious_dir, target_layer=args.layer, alpha=args.alpha
    )

    prompts = read_csv(args.input_csv)
    results = []

    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}")
            
        # Apply chat template
        tokenized_chat, op_length = apply_chat_template(prompt, tokenizer, args.device)
            
        # Generate text with activation addition
        print("Generating text with activation addition...")
        modified_text = gen_text(model, tokenizer, op_length, tokenized_chat, args.max_new_tokens)
            
        # Save result
        results.append({
            "forbidden_prompt": prompt, 
            "response": modified_text,
        })
        del prompt, modified_text
        gc.collect()
        torch.cuda.empty_cache()

    save_csv(results, args.output_csv)
    
    # Remove hooks to clean up
    for handle in addition_hooks:
        handle.remove()
    
    print("Activation addition completed successfully!")

def run():
    main()

if __name__ == "__main__":
    main()