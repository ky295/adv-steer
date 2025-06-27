import argparse
import gc
from typing import List

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m utils.batch_csv_generation --input_csv dataset/cautious_eval.csv --output_csv dataset/validate/cautious_eval_output_vllm.csv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through the model using vLLM with repetitions"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help="Load the model")
    parser.add_argument('--input_csv', type=str, default='dataset/cautious.csv',    
                        help='Path to the input CSV file with prompts')
    parser.add_argument('--output_csv', type=str, default='dataset/cautious_output_vllm2.csv',
                        help='Path to save the output CSV file')
    parser.add_argument("--max_new_tokens", type=int, default=2048, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, 
                        help="Top-p value for nucleus sampling")
    parser.add_argument("--repetitions", type=int, default=5, 
                        help="Number of times to repeat generation for each prompt")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, 
                        help="GPU memory utilization ratio")
    return parser.parse_args()

def read_csv(input_csv: str) -> List[str]:
    """Read prompts from the CSV file."""
    print(f"Reading prompts from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        prompts = df['forbidden_prompt'].tolist()
        print(f"Loaded {len(prompts)} prompts from the CSV file")
        return prompts
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return None

def save_csv(results: List[dict], output_csv: str):
    """Save results to CSV."""
    print(f"\nSaving results to {output_csv}...")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        print(f"Results saved successfully to {output_csv}")
        print(f"Total rows saved: {len(results)}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

def apply_chat_template_batch(prompts: List[str], tokenizer) -> List[str]:
    """Apply chat template to a batch of prompts."""
    formatted_prompts = []
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            chat, 
            add_generation_prompt=True, 
            tokenize=False
        )
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts

def process_batch(llm: LLM, tokenizer, prompts_batch: List[str], 
                 sampling_params: SamplingParams, repetitions: int) -> List[dict]:
    """Process a batch of prompts with repetitions."""
    results = []
    
    # Create repeated prompts for this batch
    repeated_prompts = []
    original_indices = []
    
    for i, prompt in enumerate(prompts_batch):
        for rep in range(repetitions):
            repeated_prompts.append(prompt)
            original_indices.append(i)
    
    # Apply chat template to all repeated prompts
    formatted_prompts = apply_chat_template_batch(repeated_prompts, tokenizer)
    
    print(f"Generating responses for {len(formatted_prompts)} prompts (batch size: {len(prompts_batch)}, repetitions: {repetitions})...")
    
    # Generate responses
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Process outputs and group by original prompt
    for i, output in enumerate(outputs):
        original_idx = original_indices[i]
        original_prompt = prompts_batch[original_idx]
        response = output.outputs[0].text
        repetition = i % repetitions + 1
        
        results.append({
            "forbidden_prompt": original_prompt,
            "response": response,
            "repetition": repetition
        })
    
    return results

def main():
    args = parse_args()
    
    # Read prompts
    prompts = read_csv(args.input_csv)
    if prompts is None:
        return
    
    # Initialize tokenizer for chat template
    print("Loading tokenizer for chat template...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096,  # Adjust based on your model's context length
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        # top_p=args.top_p
    )
    
    print(f"Processing {len(prompts)} prompts with {args.repetitions} repetitions each...")
    print(f"Total generations: {len(prompts) * args.repetitions}")
    
    all_results = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), args.batch_size):
        batch_end = min(i + args.batch_size, len(prompts))
        prompts_batch = prompts[i:batch_end]
        
        print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(prompts) + args.batch_size - 1)//args.batch_size}")
        print(f"Batch range: {i+1}-{batch_end} of {len(prompts)} prompts")
        
        batch_results = process_batch(llm, tokenizer, prompts_batch, sampling_params, args.repetitions)
        all_results.extend(batch_results)
        
        # Clean up memory
        gc.collect()
    
    # Save results
    save_csv(all_results, args.output_csv)
    print(f"\nCompleted! Generated {len(all_results)} total responses.")

def run():
    main()

if __name__ == "__main__":
    main()