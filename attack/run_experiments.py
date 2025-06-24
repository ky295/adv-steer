"""
A script to run evaluation experiments for GCG / GCG+IRIS attacks against Deepseek's R1 model.
Optimized version with per-worker model loading to prevent CUDA OOM.
"""

# Import nanogcg (attack) module 
import sys
sys.path.append("..")  
sys.path.append(".")
from attack import run, GCGConfig 

import argparse
import datetime
import fcntl
import gc
import multiprocessing as mp
import os
import time
import traceback
from typing import Dict, List, Tuple

import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore", message="Flash Attention defaults to a non-deterministic algorithm")

# Set HF cache location before imports
hf_cache = "/scratch/etheridge/huggingface"
os.environ["HF_HOME"] = hf_cache
os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Disable tokenizer parallelism to avoid fork warnings
)



# Configuration constants
WANDB_ENTITY = "reasoning_attacks"
WANDB_PROJECT = "gcg-iris-eval"
CAUTION_VECTOR = "../cautious_dir/cautious_dir_standard.pt"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DTYPE = "bfloat16"

NUM_GPUS = 4
RUNS_PER_GPU = 1

NUM_STEPS = 150
EARLY_STOP = False
BASE_SEED = 42

REFUSAL_LAYER_IDX = 17

RESULTS_DIR = "results/"
INPUT_CSV = "../dataset/orthogonalized_outputs_cot150_2048.csv"

# Override target string
OVERRIDE_TARGET_STRING = "</think>\n\nSure"


# Global variables for worker process
_worker_model = None
_worker_tokenizer = None
_worker_gpu_id = None


def worker_init(gpu_id: int):
    """Initialize worker process with model and tokenizer."""
    global _worker_model, _worker_tokenizer, _worker_gpu_id

    _worker_gpu_id = gpu_id

    print(f"Initializing worker process on GPU {gpu_id}...")

    try:
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)  # Use device 0 since we set CUDA_VISIBLE_DEVICES

        # Clear any existing cache
        torch.cuda.empty_cache()

        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _worker_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _worker_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=getattr(torch, DTYPE),
            device_map="cuda:0",  # Use device 0 since we limited visibility
            trust_remote_code=True,
        )

        # Monitor memory after loading
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(
            f"Worker on GPU {gpu_id}: Model loaded. Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )

    except Exception as e:
        print(f"Error initializing worker on GPU {gpu_id}: {e}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch GCG experiments")
    parser.add_argument(
        "--beta",
        type=float,
        # choices=[0.0, 0.3, 0.7, 1.0],
        required=True,
        help="Refusal beta value: 0 (token forcing only), 0.7 (mixed), 1.0 (refusal dir only)",
    )
    parser.add_argument(
        "--num-target-tokens",
        type=int,
        default=20,
        help="Number of tokens to extract from target response for token forcing (default: 20)",
    )
    parser.add_argument(
        "--num-refusal-tokens",
        type=int,
        default=45,
        help="Number of tokens to use for refusal direction computation (default: 45)",
    )
    parser.add_argument(
        "--refusal-layer-idx",
        type=int,
        default=REFUSAL_LAYER_IDX,
        help=f"Layer index for refusal vector (default: {REFUSAL_LAYER_IDX})",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=NUM_STEPS,
        help=f"Number of steps for GCG (default: {NUM_STEPS})",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=NUM_GPUS,
        help=f"Number of GPUs to use (default: {NUM_GPUS})",
    )
    parser.add_argument(
        "--runs-per-gpu",
        type=int,
        default=RUNS_PER_GPU,
        help=f"Number of runs per GPU (default: {RUNS_PER_GPU})",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=INPUT_CSV,
        help=f"Input CSV file path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print experiment plan without running"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--target-override",
        action="store_true",
        help="Override target string extraction with '</think>\\nSure, here is'",
    )
    return parser.parse_args()


def load_input_data(csv_path: str) -> pd.DataFrame | None:
    """Load input data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["forbidden_prompt", "response"]

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        df = df.dropna(subset=required_columns)
        print(f"Loaded {len(df)} experiments from {csv_path}")
        return df

    except Exception as e:
        print(f"Error loading input CSV: {e}")
        sys.exit(1)


def preprocess_target_strings(
    df: pd.DataFrame, target_tokens: int, use_override: bool = False
) -> pd.DataFrame:
    """Preprocess all target strings by tokenizing them properly or using override."""
    if use_override:
        print(f"Using override target string: '{OVERRIDE_TARGET_STRING}'")
        df["target_string"] = OVERRIDE_TARGET_STRING
        print(f"Set override target string for {len(df)} experiments")
        return df

    print("Preprocessing target strings with tokenizer...")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def extract_target_tokens(response: str, num_tokens: int) -> str:
        """Extract first n tokens from the response using actual tokenizer."""
        try:
            tokens = tokenizer.encode(response, add_special_tokens=False)
            target_tokens = tokens[:num_tokens]
            target_string = tokenizer.decode(target_tokens, skip_special_tokens=True)
            return target_string.strip()
        except Exception as e:
            print(f"Error tokenizing response: {e}")
            words = response.strip().split()[:num_tokens]
            return " ".join(words)

    df["target_string"] = df["response"].apply(
        lambda x: extract_target_tokens(x, target_tokens)
    )

    print(f"Preprocessed {len(df)} target strings")
    return df


def get_output_csv_path(
    results_dir: str,
    beta: float,
    target_tokens: int,
    refusal_tokens: int,
    use_override: bool = False,
) -> str:
    """Generate output CSV filename based on beta value and token counts."""
    beta_str = f"{beta:.1f}".replace(".", "p")
    # timestamp = "10optim"
    timestamp = ""

    if use_override:
        return os.path.join(
            results_dir,
            f"gcg_results_beta_{beta_str}_override_target_refusal_{refusal_tokens}_{timestamp}.csv",
        )
    else:
        return os.path.join(
            results_dir,
            f"gcg_results_beta_{beta_str}_target_{target_tokens}_refusal_{refusal_tokens}_{timestamp}.csv",
        )


def load_existing_results(output_csv: str) -> pd.DataFrame:
    """Load existing results to resume interrupted experiments."""
    if os.path.exists(output_csv):
        try:
            df = pd.read_csv(output_csv)
            print(f"Found existing results file with {len(df)} completed experiments")
            return df
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def setup_wandb_config(
    beta: float,
    prompt_idx: int,
    gpu_id: int,
    target_tokens: int,
    refusal_tokens: int,
    num_steps: int,
    forbidden_prompt: str,
    use_override: bool = False,
) -> Dict:
    """Set up wandb configuration for experiment tracking."""
    prompt_words = forbidden_prompt.strip().split()[:5]
    prompt_snippet = "_".join(prompt_words)

    # Clean up filename-unfriendly characters
    for char in '/\\:?*|<>"':
        prompt_snippet = prompt_snippet.replace(char, "_")

    if len(prompt_snippet) > 50:
        prompt_snippet = prompt_snippet[:50]

    if use_override:
        run_name = f"beta_{beta:.1f}_override_prompt_{prompt_idx}_{prompt_snippet}"
    else:
        run_name = f"beta_{beta:.1f}_prompt_{prompt_idx}_{prompt_snippet}"

    config = {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "name": run_name,
        "config": {
            "beta": beta,
            "prompt_idx": prompt_idx,
            "gpu_id": gpu_id,
            "num_steps": num_steps,
            "model": MODEL_NAME,
            "refusal_tokens": refusal_tokens,
            "use_target_override": use_override,
        },
    }

    if not use_override:
        config["config"]["target_tokens"] = target_tokens

    return config


def write_result_to_csv(result: Dict, output_csv: str):
    """Write a single result to CSV file using file locking for process safety."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    file_exists = os.path.exists(output_csv)
    df = pd.DataFrame([result])

    with open(output_csv, "a" if file_exists else "w", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        try:
            df.to_csv(f, index=False, header=not file_exists)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    print(f"Saved result for prompt_idx {result['prompt_idx']} to {output_csv}")


def monitor_memory(stage: str):
    """Monitor GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(
            f"GPU {_worker_gpu_id} [{stage}]: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )


def run_single_experiment(args: Tuple) -> Dict:
    """Run a single GCG experiment using cached model."""
    global _worker_model, _worker_tokenizer, _worker_gpu_id

    (
        prompt_idx,
        forbidden_prompt,
        target_string,
        beta,
        target_tokens,
        refusal_tokens,
        assigned_gpu_id,
        verbose,
        num_steps,
        refusal_layer_idx,
        output_csv,
        use_override,
    ) = args

    if _worker_model is None or _worker_tokenizer is None:
        raise RuntimeError(f"Worker model not initialized on GPU {_worker_gpu_id}")

    try:
        if verbose:
            monitor_memory(f"before_exp_{prompt_idx}")

        # Clear cache before experiment
        torch.cuda.empty_cache()

        # Set up wandb config
        wandb_config = setup_wandb_config(
            beta,
            prompt_idx,
            _worker_gpu_id,
            target_tokens,
            refusal_tokens,
            num_steps,
            forbidden_prompt,
            use_override,
        )

        # Configure GCG parameters
        use_refusal_direction = beta > 0.0
        use_token_forcing = beta < 1.0

        config = GCGConfig(
            verbosity="INFO" if verbose else "WARNING",
            early_stop=EARLY_STOP,
            num_steps=num_steps,
            wandb_config=wandb_config,
            seed=BASE_SEED + prompt_idx,
            run_id=wandb_config["name"],
            prompt_string=forbidden_prompt,
            target_string=target_string if use_token_forcing else "",
            target_no_think=False,
            # optim_str_init = "x x x x x x x x x x",
            # Refusal direction parameters
            use_refusal_direction=use_refusal_direction,
            refusal_vector_path=CAUTION_VECTOR if use_refusal_direction else None,
            refusal_layer_idx=refusal_layer_idx,
            refusal_num_tokens=refusal_tokens,
            use_extended_generation=True,
            refusal_beta=beta,
            use_refusal_beta_schedule=False,
            promote_caution=False,
            use_prefix=False,  # rather than suffix
        )

        messages = [{"role": "user", "content": forbidden_prompt}]

        print(
            f"GPU {_worker_gpu_id}: Starting experiment {prompt_idx} with beta={beta}"
        )
        if use_override:
            print(f"  Using override target: '{target_string}'")
        start_time = time.time()

        # Use cached model and tokenizer
        result = run(
            _worker_model,
            _worker_tokenizer,
            messages,
            target_string if use_token_forcing else "",
            config,
        )

        end_time = time.time()
        runtime = end_time - start_time

        # Extract results
        best_response = result.best_answer if hasattr(result, "best_answer") else ""
        best_suffix = result.best_string if hasattr(result, "best_string") else ""
        best_loss = result.best_loss if hasattr(result, "best_loss") else float("inf")
        all_losses = result.losses if hasattr(result, "losses") else []
        all_strings = result.strings if hasattr(result, "strings") else []

        experiment_result = {
            "prompt_idx": prompt_idx,
            "forbidden_prompt": forbidden_prompt,
            "target_string": target_string,
            "beta": beta,
            "target_tokens": target_tokens if not use_override else "override",
            "refusal_tokens": refusal_tokens,
            "use_target_override": use_override,
            "gpu_id": _worker_gpu_id,
            "best_response": best_response,
            "best_suffix": best_suffix,
            "best_loss": float(best_loss),
            "final_loss": float(all_losses[-1]) if all_losses else float("inf"),
            "initial_loss": float(all_losses[0]) if all_losses else float("inf"),
            "loss_improvement": float(all_losses[0] - all_losses[-1])
            if len(all_losses) > 1
            else 0.0,
            "num_loss_points": len(all_losses),
            "runtime_seconds": runtime,
            "num_steps": num_steps,
            "success": True,
            "error": None,
            "timestamp": datetime.datetime.now().isoformat(),
            "loss_trajectory": ",".join(map(str, all_losses)) if all_losses else "",
            "optimization_strings_sample": "|".join(all_strings[:5])
            if all_strings
            else "",
        }

        write_result_to_csv(experiment_result, output_csv)

        if verbose:
            monitor_memory(f"after_exp_{prompt_idx}")

        # Light cleanup
        torch.cuda.empty_cache()

        print(
            f"GPU {_worker_gpu_id}: Completed experiment {prompt_idx} in {runtime:.1f}s"
        )
        return experiment_result

    except torch.cuda.OutOfMemoryError as e:
        error_msg = (
            f"CUDA OOM on GPU {_worker_gpu_id} for experiment {prompt_idx}: {str(e)}"
        )
        print(error_msg)

        # Try to recover
        torch.cuda.empty_cache()
        gc.collect()

        error_result = {
            "prompt_idx": prompt_idx,
            "forbidden_prompt": forbidden_prompt,
            "target_string": target_string,
            "beta": beta,
            "target_tokens": target_tokens if not use_override else "override",
            "refusal_tokens": refusal_tokens,
            "use_target_override": use_override,
            "gpu_id": _worker_gpu_id,
            "best_response": "",
            "best_suffix": "",
            "best_loss": float("inf"),
            "runtime_seconds": 0,
            "num_steps": num_steps,
            "success": False,
            "error": error_msg,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        write_result_to_csv(error_result, output_csv)
        return error_result

    except Exception as e:
        error_msg = (
            f"Error in experiment {prompt_idx}: {str(e)}\n{traceback.format_exc()}"
        )
        print(f"GPU {_worker_gpu_id}: {error_msg}")

        error_result = {
            "prompt_idx": prompt_idx,
            "forbidden_prompt": forbidden_prompt,
            "target_string": target_string,
            "beta": beta,
            "target_tokens": target_tokens if not use_override else "override",
            "refusal_tokens": refusal_tokens,
            "use_target_override": use_override,
            "gpu_id": _worker_gpu_id,
            "best_response": "",
            "best_suffix": "",
            "best_loss": float("inf"),
            "runtime_seconds": 0,
            "num_steps": num_steps,
            "success": False,
            "error": error_msg,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        write_result_to_csv(error_result, output_csv)
        return error_result


def prepare_experiments(
    df: pd.DataFrame,
    beta: float,
    target_tokens: int,
    refusal_tokens: int,
    existing_results: pd.DataFrame,
    verbose: bool,
    num_gpus: int,
    num_steps: int,
    refusal_layer_idx: int,
    output_csv: str,
    use_override: bool,
) -> List[Tuple]:
    """Prepare list of experiments to run, excluding already completed ones."""

    experiments = []
    completed_indices = (
        set(existing_results["prompt_idx"].tolist())
        if not existing_results.empty
        else set()
    )

    for idx, row in df.iterrows():
        if idx in completed_indices:
            print(f"Skipping experiment {idx} (already completed)")
            continue

        assigned_gpu_id = len(experiments) % num_gpus

        experiment_args = (
            idx,
            row["forbidden_prompt"],
            row["target_string"],
            beta,
            target_tokens,
            refusal_tokens,
            assigned_gpu_id,
            verbose,
            num_steps,
            refusal_layer_idx,
            output_csv,
            use_override,
        )
        experiments.append(experiment_args)

    return experiments


def run_experiments_on_gpu(gpu_id: int, experiments: List[Tuple]) -> List[Dict]:
    """Run a batch of experiments on a specific GPU."""
    # Initialize worker for this GPU
    worker_init(gpu_id)

    results = []
    for experiment in experiments:
        try:
            result = run_single_experiment(experiment)
            results.append(result)
        except Exception as e:
            print(f"Error running experiment on GPU {gpu_id}: {e}")
            # Create error result
            prompt_idx = experiment[0]
            error_result = {
                "prompt_idx": prompt_idx,
                "success": False,
                "error": str(e),
                "gpu_id": gpu_id,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            results.append(error_result)

    return results


def main():
    """Main execution function."""
    args = parse_args()

    print("\n=== Running GCG Experiments ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("==============================\n")

    os.makedirs(args.results_dir, exist_ok=True)

    # Load and preprocess data
    input_df = load_input_data(args.input_csv)
    input_df = preprocess_target_strings(
        input_df, args.num_target_tokens, args.target_override
    )

    output_csv = get_output_csv_path(
        args.results_dir,
        args.beta,
        args.num_target_tokens,
        args.num_refusal_tokens,
        args.target_override,
    )
    existing_results = load_existing_results(output_csv)

    experiments = prepare_experiments(
        input_df,
        args.beta,
        args.num_target_tokens,
        args.num_refusal_tokens,
        existing_results,
        args.verbose,
        args.num_gpus,
        args.num_steps,
        args.refusal_layer_idx,
        output_csv,
        args.target_override,
    )

    if not experiments:
        print("No experiments to run (all already completed)")
        return

    target_info = (
        f"override target ('{OVERRIDE_TARGET_STRING}')"
        if args.target_override
        else f"{args.num_target_tokens} target tokens"
    )
    print(f"Planning to run {len(experiments)} experiments with beta={args.beta}")
    print(
        f"Using {args.num_gpus} GPUs, {target_info}, {args.num_refusal_tokens} refusal tokens"
    )
    print(f"Results will be saved incrementally to: {output_csv}")

    if args.dry_run:
        print("Dry run - not executing experiments")
        for i, exp in enumerate(experiments[:5]):
            print(f"  Experiment {i}: prompt_idx={exp[0]}, assigned_gpu={exp[6]}")
        if len(experiments) > 5:
            print(f"  ... and {len(experiments) - 5} more")
        return

    # Distribute experiments across GPUs
    experiments_per_gpu = [[] for _ in range(args.num_gpus)]
    for i, experiment in enumerate(experiments):
        gpu_id = i % args.num_gpus
        experiments_per_gpu[gpu_id].append(experiment)

    print(f"Starting {len(experiments)} experiments across {args.num_gpus} GPUs...")
    for gpu_id in range(args.num_gpus):
        print(f"  GPU {gpu_id}: {len(experiments_per_gpu[gpu_id])} experiments")

    start_time = time.time()

    # Create process pool with one process per GPU
    with mp.Pool(processes=args.num_gpus) as pool:
        # Map each GPU to its batch of experiments
        gpu_experiment_pairs = [
            (gpu_id, experiments_per_gpu[gpu_id])
            for gpu_id in range(args.num_gpus)
            if experiments_per_gpu[gpu_id]
        ]  # Only include GPUs with experiments

        # Run experiments in parallel across GPUs
        all_results = pool.starmap(run_experiments_on_gpu, gpu_experiment_pairs)

        # Flatten results
        results = []
        for gpu_results in all_results:
            results.extend(gpu_results)

    end_time = time.time()
    total_time = end_time - start_time

    # Print summary
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful

    print("\n=== Experiment Summary ===")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f} seconds")
    if len(results) > 0:
        print(f"Average time per experiment: {total_time / len(results):.1f} seconds")
    print(f"All results saved to: {output_csv}")

    if failed > 0:
        print(f"\nErrors occurred in {failed} experiments. Check the CSV for details.")

    # Sort final results
    try:
        final_df = pd.read_csv(output_csv)
        final_df = final_df.sort_values("prompt_idx")
        final_df.to_csv(output_csv, index=False)
        print(
            f"Final results file sorted and saved with {len(final_df)} total experiments"
        )
    except Exception as e:
        print(f"Warning: Could not sort final results file: {e}")


if __name__ == "__main__":
    main()
