import argparse
import functools
import gc
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.dir_ablation_allcot --layer 17


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
    parser.add_argument('--output_csv', type=str, default='dataset/directional_ablation_allcot.csv',
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

def find_think_close_position(tokenizer, generated_ids, prompt_length):
    """
    Find the position of </think> tag in the generated sequence.
    
    Args:
        tokenizer: The tokenizer used
        generated_ids: The full generated token sequence
        prompt_length: Length of the original prompt tokens
    
    Returns:
        Position of </think> tag, or None if not found
    """
    # Get the token ID for </think>
    think_close_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    
    # Search for the pattern in generated tokens (after prompt)
    generated_tokens = generated_ids[prompt_length:]
    
    # Convert to list for easier searching
    token_list = generated_tokens.tolist() if torch.is_tensor(generated_tokens) else generated_tokens
    
    # Look for the </think> pattern
    for i in range(len(token_list) - len(think_close_tokens) + 1):
        if token_list[i:i+len(think_close_tokens)] == think_close_tokens:
            # Return absolute position (including prompt)
            absolute_pos = prompt_length + i + len(think_close_tokens) - 1  # -1 to include the last token of </think>
            return absolute_pos
    
    return None

# Global variables to track ablation range
class AblationTracker:
    def __init__(self):
        self.start_pos = None
        self.end_pos = None
        self.think_close_tokens = None
        self.found_think_close = False
    
    def reset(self, prompt_length, tokenizer):
        """Reset tracking variables for each new generation"""
        self.start_pos = prompt_length
        self.end_pos = None
        self.think_close_tokens = tokenizer.encode("</think>", add_special_tokens=False)
        self.found_think_close = False
        print(f"Initialized ablation tracker: start_pos={self.start_pos}, </think> tokens={self.think_close_tokens}")
    
    def update_end_pos_if_found(self, current_ids, tokenizer):
        """Check if </think> appears in current sequence and update end position"""
        if self.found_think_close or self.end_pos is not None:
            return
        
        # Convert to list for searching
        if torch.is_tensor(current_ids):
            token_list = current_ids.tolist()
        else:
            token_list = current_ids
        
        # Search for </think> pattern starting from start_pos
        search_start = max(0, self.start_pos)
        for i in range(search_start, len(token_list) - len(self.think_close_tokens) + 1):
            if token_list[i:i+len(self.think_close_tokens)] == self.think_close_tokens:
                # Found </think>, set end position to include the closing tag
                self.end_pos = i + len(self.think_close_tokens)
                self.found_think_close = True
                print(f"Found </think> at position {i}, setting end_pos to {self.end_pos}")
                break

# Global tracker instance
ablation_tracker = AblationTracker()

# Directional ablation hook function with CoT-specific targeting
def directional_ablation_hook_cot(
    module: torch.nn.Module,
    input: List[torch.Tensor],
    output,
    direction: torch.Tensor
):
    """
    Hook function that applies directional ablation only to chain-of-thought tokens.
    
    Args:
        module: The module whose forward pass is being hooked
        input: Inputs to the module
        output: Output of the module (can be tuple or tensor)
        direction: The normalized direction to ablate (unit vector)
    
    Returns:
        Modified output with the direction ablated only for CoT tokens
    """
    hidden_states = output[0]
    other_outputs = output[1:] if len(output) > 1 else ()
    
    # Get original shape
    original_shape = hidden_states.shape
    batch_size, seq_len, hidden_dim = original_shape
    
    # Only proceed if we have a valid ablation range
    if ablation_tracker.start_pos is None:
        return output
    
    # Try to detect </think> in the current sequence if not found yet
    if not ablation_tracker.found_think_close and hasattr(input[0], 'input_ids'):
        # Get the current full sequence from the input
        current_sequence = input[0].input_ids if hasattr(input[0], 'input_ids') else input[0]
        if current_sequence is not None:
            ablation_tracker.update_end_pos_if_found(current_sequence[0], None)
    
    # Create a mask for which positions to ablate
    ablate_mask = torch.zeros(seq_len, dtype=torch.bool, device=hidden_states.device)
    
    # Determine ablation range
    start_pos = ablation_tracker.start_pos
    end_pos = ablation_tracker.end_pos if ablation_tracker.end_pos is not None else seq_len
    
    # Only ablate tokens between prompt end and </think> (or current position if </think> not found yet)
    if start_pos < seq_len:
        actual_end = min(end_pos, seq_len)
        if actual_end > start_pos:
            ablate_mask[start_pos:actual_end] = True
    
    # Apply ablation only to masked positions
    if ablate_mask.any():
        # Reshape to 2D for easier computation
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Create position mask for flattened tensor
        pos_mask = ablate_mask.unsqueeze(0).expand(batch_size, -1).flatten()
        
        # Apply directional ablation only to masked positions
        if pos_mask.any():
            masked_states = hidden_states_flat[pos_mask]
            projections = torch.matmul(masked_states, direction)
            ablated_masked = masked_states - torch.outer(projections, direction)
            hidden_states_flat[pos_mask] = ablated_masked
        
        # Reshape back to original shape
        hidden_states = hidden_states_flat.view(batch_size, seq_len, hidden_dim)
    
    return (hidden_states,) + other_outputs

def register_directional_ablation_hooks_cot(model, direction):
    """
    Register hooks for directional ablation on chain-of-thought tokens only.
    
    Args:
        model: The model to add hooks to
        direction: The normalized direction to ablate
    
    Returns:
        handles: List of hook handles for removal later
    """
    handles = []
    hook_fn = functools.partial(
        directional_ablation_hook_cot, 
        direction=direction
    )
    
    # Hook ALL transformer layers
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(hook_fn))
    
    print(f"Registered CoT-specific directional ablation hooks on {len(handles)} layers")
    print(f"Direction norm: {direction.norm().item():.6f} (should be ~1.0 for normalized)")
    return handles

def gen_text_with_cot_ablation(model, tokenizer, op_length, tokenized_chat, max_new_tokens, direction):
    """Generate text with CoT-specific directional ablation"""
    
    # Reset tracking for this generation
    ablation_tracker.reset(op_length, tokenizer)
    
    # Register hooks with CoT-specific logic
    ablation_hooks = register_directional_ablation_hooks_cot(model, direction)
    
    try:
        with torch.no_grad():
            # Custom generation loop to track </think> detection
            input_ids = tokenized_chat
            generated_tokens = 0
            
            for step in range(max_new_tokens):
                # Forward pass
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Sample next token
                probs = torch.softmax(next_token_logits / 0.6, dim=-1)  # temperature=0.6
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated_tokens += 1
                
                # Check if we found </think> in the current sequence
                ablation_tracker.update_end_pos_if_found(input_ids[0], tokenizer)
                
                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Decode the generated output
            response = tokenizer.decode(input_ids[0][op_length:], skip_special_tokens=True)
            
            # Sanity checks
            print("\n=== SANITY CHECKS ===")
            print(f"Total generated tokens: {generated_tokens}")
            print(f"Prompt length: {op_length}")
            print(f"Final ablation range: {ablation_tracker.start_pos} to {ablation_tracker.end_pos}")
            
            # Check if </think> was found
            if ablation_tracker.found_think_close:
                print(f"</think> found at position: {ablation_tracker.end_pos - len(ablation_tracker.think_close_tokens)}")
                print(f"Relative position in generation: {ablation_tracker.end_pos - len(ablation_tracker.think_close_tokens) - op_length}")
                
                # Show tokens around </think>
                think_start = ablation_tracker.end_pos - len(ablation_tracker.think_close_tokens)
                context_start = max(0, think_start - 5)
                context_end = min(len(input_ids[0]), think_start + 10)
                context_tokens = input_ids[0][context_start:context_end]
                context_text = tokenizer.decode(context_tokens, skip_special_tokens=False)
                print(f"Context around </think>: {repr(context_text)}")
            else:
                print("WARNING: </think> tag not found in generation!")
            
            # Check if ablation was applied
            if ablation_tracker.start_pos is not None and ablation_tracker.end_pos is not None:
                ablated_tokens = ablation_tracker.end_pos - ablation_tracker.start_pos
                print(f"Ablated {ablated_tokens} CoT tokens")
            elif ablation_tracker.start_pos is not None:
                ablated_tokens = generated_tokens
                print(f"Ablated approximately {ablated_tokens} CoT tokens (</think> not found)")
            else:
                print("WARNING: No ablation range was set!")
            
            print("=== END SANITY CHECKS ===\n")
    
    finally:
        # Remove hooks to clean up
        for handle in ablation_hooks:
            handle.remove()
    
    return response

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Using CoT-specific directional ablation across ALL layers")
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

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name, args.device)

    # Test tokenizer with </think> tag
    think_close_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    print(f"</think> tokenizes to: {think_close_tokens}")
    print(f"</think> decoded back: {repr(tokenizer.decode(think_close_tokens))}")

    prompts = read_csv(args.input_csv)
    results = []

    # Process each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}")
            
        # Apply chat template
        tokenized_chat, op_length = apply_chat_template(prompt, tokenizer, args.device)
        
        # Show prompt tokens for debugging
        prompt_text = tokenizer.decode(tokenized_chat[0], skip_special_tokens=False)
        print(f"Prompt tokens (length {op_length}): {repr(prompt_text[:200])}...")
            
        # Generate text with CoT-specific directional ablation
        print("Generating text with CoT-specific directional ablation...")
        ablated_text = gen_text_with_cot_ablation(
            model, tokenizer, op_length, tokenized_chat, args.max_new_tokens, cautious_dir
        )
            
        # Save result
        results.append({
            "forbidden_prompt": prompt, 
            "response": ablated_text,
        })
        del prompt, ablated_text
        gc.collect()
        torch.cuda.empty_cache()

    save_csv(results, args.output_csv)
    
    print("CoT-specific directional ablation completed successfully!")

def run():
    main()

if __name__ == "__main__":
    main()