import atexit
import copy
import gc
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from torch import Tensor
from tqdm import tqdm
from transformers import set_seed
from transformers.cache_utils import DynamicCache

import wandb
from attack.utils import (INIT_CHARS, configure_pad_token,
                           find_executable_batch_size, get_nonascii_toks,
                           mellowmax)



logger: logging.Logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProbeSamplingConfig:
    """
    Configuration parameters for the probe sampling attack.
    """
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16


@dataclass
class GCGConfig:
    """
    Configuration parameters for the GCG+CoT refusal minimisation attack.
    """
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: Optional[int] = None  
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = 42
    verbosity: str = "INFO"
    probe_sampling_config: Optional[ProbeSamplingConfig] = None
    wandb_config: dict = field(default_factory=dict)
    run_id: Optional[str] = None
    prompt_string: Optional[str] = None
    target_string: Optional[str] = None
    target_no_think: Optional[bool] = None
    max_new_tokens: int = 2048  # Maximum new tokens to generate in the final response

    # Generation parameters for sampling (defaults from your experiments)
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95

    # Adversarial positioning options
    use_prefix: bool = False  # If True, use adversarial prefix instead of suffix
    example_generation_frequency: int = (
        25  # How often to generate examples during optimization
    )

    # refusal direction minimisation terms
    use_refusal_direction: bool = False
    refusal_vector_path: Optional[str] = None
    refusal_vector: Optional[torch.Tensor] = None
    refusal_layer_idx: Union[int, List[int]] = (
        17  # Single layer (int), list of layers, or empty list for all layers
    )
    refusal_num_tokens: int = 10  # Number of CoT tokens to use for refusal calculation
    refusal_beta: float = 0.75  # weight balancing term
    promote_caution: bool = (
        False  # If True, maximize refusal alignment instead of minimizing
    )

    # Extended generation for refusal calculation
    refusal_generation_length: int = (
        50  # How many additional tokens to generate for refusal calculation
    )
    use_extended_generation: bool = (
        True  # Whether to generate additional tokens beyond target for refusal calc
    )

    # Option to average across all layers instead of using specific layer
    refusal_average_all_layers: bool = (
        False  # DEPRECATED: Use refusal_layer_idx=[] instead
    )

    # CoT detection tokens (DeepSeek specific by default)
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"

    # scheduling parameters, to balance loss terms with a variable beta
    use_refusal_beta_schedule: bool = False
    schedule_type: str = "linear"  # "linear" or "cosine"
    refusal_beta_start: float = 0.0
    refusal_beta_end: float = 0.75
    schedule_start_step: int = 0
    schedule_end_step: Optional[int] = None  # If None, will use num_steps
    schedule_exponent: float = 2.0  # For exponential schedule
    schedule_power: float = 2.0  # For power schedule


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    best_answer: Optional[str] = None


class AttackBuffer:
    """A buffer to store the best optimization results."""
    def __init__(self, size: int) -> None:
        """
        Initializes the attack buffer.
        Args:
            size: Maximum number of elements to store in the buffer.
        """
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        """
        Adds a new optimization result to the buffer.
        Args:
            loss: The loss value associated with the optimization result.
            optim_ids: The optimized token IDs as a Tensor.
        """
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        """
        Returns the optimized token IDs with the lowest loss.
        Returns:
            Tensor of optimized token IDs with the lowest loss.
        """
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        """
        Returns:
            The lowest loss value in the buffer as a float.
        """
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        """
        Returns:
            The highest loss value in the buffer as a float.
        """
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        """
        Logs the contents of the buffer to the logger.
        Args:
            tokenizer: The tokenizer used to decode the optimized token IDs.
        """
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)




def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Optional[Tensor] = None
) -> Tensor:
    """Samples candidate token sequences based on gradient information.

    Args:
        ids: Current optimized token IDs of shape [1, seq_len].
        grad: Token gradient of shape [1, seq_len, vocab_size].
        search_width: Number of candidate sequences to generate.
        topk: Number of top tokens to consider for replacement.
        n_replace: Number of tokens to replace per candidate.
        not_allowed_ids: Token IDs to exclude from sampling.

    Returns:
        Tensor of candidate token sequences of shape [search_width, seq_len].
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(
        torch.rand((search_width, n_optim_tokens), device=grad.device)
    )[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer) -> Tensor:
    """Filters token sequences that change after retokenization.

    Args:
        ids: Token ID sequences to filter.
        tokenizer: Tokenizer for encoding/decoding.

    Returns:
        Tensor of filtered token sequences that remain consistent after retokenization.
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(
            ids_decoded[i], return_tensors="pt", add_special_tokens=False
        ).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids)


class GCG:
    """Greedy Coordinate Gradient attack implementation with refusal direction minimisation term."""
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        """Initialise GCG optimizer.

        Args:
            model: Target model for optimization.
            tokenizer: Model's tokenizer.
            config: GCG configuration parameters.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings() # type: ignore
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )
        self.prefix_cache = None
        self.draft_prefix_cache = None

        self.stop_flag = False
        self.step = 0  # Initialize step counter
        self.wandb_metrics = {}  # Initialize metrics dictionary

        # Storage for activations during forward pass
        self.layer_activations = {}
        self.activation_hooks = []

        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_embedding_layer = None
        if self.config.probe_sampling_config:
            self.draft_model = self.config.probe_sampling_config.draft_model
            self.draft_tokenizer = self.config.probe_sampling_config.draft_tokenizer
            self.draft_embedding_layer = self.draft_model.get_input_embeddings()
            if self.draft_tokenizer.pad_token is None:
                configure_pad_token(self.draft_tokenizer)

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )

        if model.device == torch.device("cpu"):
            logger.warning(
                "Model is on the CPU. Use a hardware accelerator for faster optimization."
            )

        if not tokenizer.chat_template:
            logger.warning(
                "Tokenizer does not have a chat template. Assuming base model and setting chat template to empty."
            )
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )

        # Initialize refusal direction if enabled
        if config.use_refusal_direction:
            self._setup_refusal_direction()

        if config.wandb_config:
            # if old notebook wandb run found, clear it
            if wandb.run is not None:
                wandb.finish()
                logger.info("Cleared old wandb run.")

            self.using_wandb = True

            wandb.init(
                project=config.wandb_config["project"],
                entity=config.wandb_config["entity"],
                config=asdict(config),
            )


            if wandb.run is not None:
                wandb.run.name = config.wandb_config["name"] # type: ignore
                logger.info(f"Initialized wandb run: {wandb.run.name}") # type: ignore

        else:
            self.using_wandb = False
            logger.info("Wandb not initialized.")

        atexit.register(self._cleanup)

    def _cleanup(self):
        if self.using_wandb:
            wandb.finish()
            logger.info("Wandb run finished.")
        self._remove_activation_hooks()

    def _setup_refusal_direction(self):
        """Initialize refusal direction vectors and hooks."""
        config = self.config

        if config.refusal_vector is not None:
            self.refusal_vector = config.refusal_vector.to(
                self.model.device, self.model.dtype
            )
            logger.info(
                f"Using provided refusal vector with shape {self.refusal_vector.shape}"
            )
        elif config.refusal_vector_path:
            try:
                # Try loading with torch
                self.refusal_vector = torch.load(
                    config.refusal_vector_path, map_location=self.model.device
                )
                logger.info(
                    f"Loaded refusal vector from {config.refusal_vector_path} using torch.load"
                )
            # except not being able to load into torch
            except (FileNotFoundError, torch.serialization.pickle.UnpicklingError, IsADirectoryError, EOFError):
                # Fall back to numpy
                try:
                    refusal_array = np.load(config.refusal_vector_path)
                    self.refusal_vector = torch.tensor(
                        refusal_array, device=self.model.device
                    )
                    logger.info(
                        f"Loaded refusal vector from {config.refusal_vector_path} using numpy"
                    )
                except Exception as e:
                    logger.error(f"Failed to load refusal vector: {e}")
                    raise

            # Convert to model's data type
            self.refusal_vector = self.refusal_vector.to(dtype=self.model.dtype)
            logger.info(
                f"Loaded refusal vector with shape {self.refusal_vector.shape} and dtype {self.refusal_vector.dtype}"
            )

        else:
            raise ValueError(
                "When use_refusal_direction=True, must provide either refusal_vector or refusal_vector_path"
            )

        # Normalize the refusal vector for stable dot product calculations
        self.refusal_vector = self.refusal_vector / torch.norm(self.refusal_vector)

        # Get CoT detection token IDs
        self.think_start_id = self.tokenizer.encode(
            config.think_start_token, add_special_tokens=False
        )[0]
        self.think_end_id = self.tokenizer.encode(
            config.think_end_token, add_special_tokens=False
        )[0]

        # Register activation hooks
        self._register_activation_hooks()
        logger.info("Registered activation hooks for refusal direction")

    def _register_activation_hooks(self):
        """Register hooks to capture layer activations."""

        def get_activation_hook(layer_idx):
            def hook(module, input, output):
                # Store the input to the layer (residual stream)
                if isinstance(input, tuple):
                    activation = input[0]
                else:
                    activation = input
                self.layer_activations[layer_idx] = activation

            return hook

        # Clear existing hooks
        self._remove_activation_hooks()

        # Determine which layers to hook based on refusal_layer_idx
        if isinstance(self.config.refusal_layer_idx, int):
            # Single layer
            target_layers = [self.config.refusal_layer_idx]
        elif isinstance(self.config.refusal_layer_idx, list):
            if len(self.config.refusal_layer_idx) == 0:
                # Empty list means all layers
                if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    target_layers = list(range(len(self.model.model.layers)))
                else:
                    logger.warning(
                        "Cannot determine model layers for empty refusal_layer_idx list"
                    )
                    target_layers = []
            else:
                # Specific list of layers
                target_layers = self.config.refusal_layer_idx
        else:
            logger.error(
                f"Invalid refusal_layer_idx type: {type(self.config.refusal_layer_idx)}"
            )
            target_layers = []

        # Register hooks for target layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            total_layers = len(self.model.model.layers)

            for layer_idx in target_layers:
                if 0 <= layer_idx < total_layers:
                    layer = self.model.model.layers[layer_idx]
                    if hasattr(layer, "input_layernorm"):
                        hook = layer.input_layernorm.register_forward_hook(
                            get_activation_hook(layer_idx)
                        )
                        self.activation_hooks.append(hook)
                    else:
                        logger.warning(
                            f"Layer {layer_idx} doesn't have input_layernorm"
                        )
                else:
                    logger.warning(
                        f"Layer index {layer_idx} out of bounds (model has {total_layers} layers)"
                    )

            logger.info(
                f"Registered hooks for layers: {[idx for idx in target_layers if 0 <= idx < total_layers]}"
            )
        else:
            logger.error("Cannot access model layers for hook registration")

    def _remove_activation_hooks(self):
        """Remove ALL activation hooks."""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        self.layer_activations = {}

    def _find_cot_token_positions(
        self, input_ids: Tensor
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the positions of CoT start and end tokens in the sequence.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len] or [seq_len]

        Returns:
            Tuple of (cot_start_pos, cot_end_pos) or (None, None) if not found
        """
        if input_ids.dim() > 1:
            input_ids = input_ids[0]  # Take first batch element

        # Find think start token
        think_start_positions = (input_ids == self.think_start_id).nonzero(
            as_tuple=True
        )[0]
        if len(think_start_positions) == 0:
            return None, None

        cot_start_pos = think_start_positions[0].item() + 1  # Position after <think>

        # Find think end token
        think_end_positions = (input_ids == self.think_end_id).nonzero(as_tuple=True)[0]
        if len(think_end_positions) == 0:
            # If no end token found, use end of sequence
            cot_end_pos = input_ids.shape[0]
        else:
            cot_end_pos = think_end_positions[0].item()  # Position of </think>

        return cot_start_pos, cot_end_pos

    def _compute_cot_refusal_loss(self, input_embeds: Tensor) -> Optional[Tensor]:
        """
        Compute refusal loss based on target tokens + additional generated tokens up to refusal_num_tokens.

        Args:
            input_embeds: Input embeddings [batch_size, seq_len, hidden_dim]
                         (may include additional generated tokens beyond target)

        Returns:
            Refusal loss term or None if CoT tokens not found
        """
        if not self.config.use_refusal_direction or not self.layer_activations:
            return None

        seq_len = input_embeds.shape[1]
        target_length = self.target_ids.shape[1]

        # Calculate the base sequence length (without any additional generated tokens)
        if self.prefix_cache:
            # Prefix cache case: [optim] + [after] + [target]
            # If we have the stored original target embeddings, we can calculate the base
            original_base_length = self.after_embeds.shape[1] + target_length
            # Additional tokens = current length - original base length
            additional_tokens_available = max(0, seq_len - original_base_length)
        else:
            # Non-prefix case: [before] + [optim] + [after] + [target]
            original_base_length = (
                self.before_embeds.shape[1] + self.after_embeds.shape[1] + target_length
            )
            additional_tokens_available = max(0, seq_len - original_base_length)

        # Find where the target sequence starts
        target_start_pos = seq_len - target_length - additional_tokens_available

        # Calculate how many tokens we can use for refusal calculation
        total_available_tokens = target_length + additional_tokens_available
        actual_tokens_to_use = min(
            self.config.refusal_num_tokens, total_available_tokens
        )

        # Debug logging
        if self.using_wandb and self.step % 10 == 0:
            self.add_wandb_metric("debug/seq_len", seq_len)
            self.add_wandb_metric("debug/target_length", target_length)
            self.add_wandb_metric(
                "debug/additional_tokens_available", additional_tokens_available
            )
            self.add_wandb_metric(
                "debug/total_available_tokens", total_available_tokens
            )
            self.add_wandb_metric("debug/target_start_pos", target_start_pos)
            self.add_wandb_metric(
                "debug/requested_refusal_tokens", self.config.refusal_num_tokens
            )
            self.add_wandb_metric("debug/actual_tokens_to_use", actual_tokens_to_use)
            self.add_wandb_metric(
                "debug/using_prefix_cache", self.prefix_cache is not None
            )

        if actual_tokens_to_use <= 0:
            logger.warning("No tokens available for refusal loss computation")
            return None

        if actual_tokens_to_use < self.config.refusal_num_tokens and self.step == 0:
            logger.info(
                f"Requested {self.config.refusal_num_tokens} refusal tokens, using {actual_tokens_to_use} (target: {target_length}, additional: {additional_tokens_available})"
            )

        # Look for CoT tokens in the activations
        total_loss: float = 0.0
        valid_layers: int = 0

        for layer_idx, activation in self.layer_activations.items():
            if activation.shape[1] != seq_len:
                logger.warning(
                    f"Layer {layer_idx}: activation length {activation.shape[1]} != seq_len {seq_len}"
                )
                continue

            # Extract the refusal region starting from target tokens
            refusal_end_pos = target_start_pos + actual_tokens_to_use
            refusal_region = activation[:, target_start_pos:refusal_end_pos]

            if refusal_region.shape[1] != actual_tokens_to_use:
                logger.warning(
                    f"Layer {layer_idx}: Expected {actual_tokens_to_use} tokens, got {refusal_region.shape[1]} (start: {target_start_pos}, end: {refusal_end_pos})"
                )
                continue

            # Compute dot products for each token and each batch element
            # refusal_region: [batch_size, actual_tokens_to_use, hidden_dim]
            # refusal_vector: [hidden_dim]
            dot_products: Tensor = torch.matmul(
                refusal_region, self.refusal_vector
            )  # [batch_size, actual_tokens_to_use]

            # Square and average across tokens
            squared_dots: Tensor = dot_products.pow(2)  # [batch_size, actual_tokens_to_use]
            layer_loss: Tensor = squared_dots.mean(dim=1)  # [batch_size]

            total_loss: Tensor = total_loss + layer_loss
            valid_layers += 1

            # Debug logging for first layer
            if (
                self.using_wandb
                and layer_idx == list(self.layer_activations.keys())[0]
                and self.step % 10 == 0
            ):
                self.add_wandb_metric(
                    "debug/refusal_region_shape_1", refusal_region.shape[1]
                )
                self.add_wandb_metric(
                    "debug/dot_products_mean", dot_products.mean().item()
                )
                self.add_wandb_metric(
                    "debug/dot_products_std", dot_products.std().item()
                )
                self.add_wandb_metric("debug/refusal_start_pos", target_start_pos)
                self.add_wandb_metric("debug/refusal_end_pos", refusal_end_pos)

        if valid_layers > 0:
            # Average across layers
            # final_loss = total_loss / valid_layers
            final_loss: Tensor = torch.div(total_loss, valid_layers)

            if self.using_wandb and self.step % 10 == 0:
                self.add_wandb_metric("debug/valid_layers_count", valid_layers)
                self.add_wandb_metric(
                    "debug/final_refusal_loss", final_loss.mean().item()
                )

            return final_loss

        logger.warning("No valid layers found for refusal loss computation")
        return None

    def add_wandb_metric(self, name: str, value: Any) -> None:
        """Add metric to wandb logging queue.
        
        Args:
            name: Metric name.
            value: Metric value.
        """
        if self.using_wandb:
            self.wandb_metrics[name] = value

    def log_wandb_metrics(self) -> None:
        """Log all queued metrics to wandb and clear the queue.
        ( we do this way to correctly align different metrics with timesteps )
        """
        if self.using_wandb and self.wandb_metrics:
            # Add step to metrics
            self.wandb_metrics["step"] = self.step

            # Log all metrics at once
            wandb.log(self.wandb_metrics)

            # Clear metrics dictionary for next step
            self.wandb_metrics = {}

    def generate_response(self, prompt: str) -> str:
        """Generate model response using configured sampling parameters.

        Args:
            prompt: Input prompt for generation (without any special tokens or dictionary/JSON formatting).

        Returns:
            Generated response text, up to `max_new_tokens` tokens and with sampling parameters specified in the config dataclass.
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        input_tensor: Tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device, torch.int64)

        attention_mask: Tensor = torch.ones_like(input_tensor)

        output: Tensor = self.model.generate(
            input_tensor,
            attention_mask=attention_mask,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response: str = self.tokenizer.batch_decode(
            output[:, input_tensor.shape[1] :], skip_special_tokens=True
        )[0]

        return response

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        """
        Run optimisation process to produce adversarial suffix, using a mixture of conventional token forcing and refusal direction minimisation.

        Args:
            messages: Input messages for the model, either as a string or a list of dictionaries with "role" and "content" keys.
            target: Target string to optimize towards

        Returns:
            GCGResult: Object containing the best loss, best string, and lists of losses and strings from each optimization step.
        """
        model: transformers.PreTrainedModel = self.model
        tokenizer: transformers.PreTrainedTokenizer = self.tokenizer
        config: GCGConfig = self.config

        if config.seed is not None:
            set_seed(config.seed)
            # torch.use_deterministic_algorithms(True, warn_only=True)
            torch.use_deterministic_algorithms(False) # set to True for fuller reproducability - a lot of noisy errors though

        if isinstance(messages, str):
            messages: list[str] = [{"role": "user", "content": messages}]
        else:
            messages: list[dict[str, str]] = copy.deepcopy(messages)

        # Append the GCG string at the appropriate location
        if not any(["{optim_str}" in d["content"] for d in messages]):
            if config.use_prefix:
                # Add as prefix: {optim_str} + original_content
                messages[-1]["content"] = "{optim_str}" + messages[-1]["content"]
            else:
                # Add as suffix: original_content + {optim_str} (original behavior)
                messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template: str = template.replace(tokenizer.bos_token, "")

        before_str: str
        after_str: str
        before_str, after_str = template.split("{optim_str}")

        target: str = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids: Tensor = tokenizer([before_str], padding=False, return_tensors="pt")[
            "input_ids"
        ].to(model.device, torch.int64)
        after_ids: Tensor = tokenizer(
            [after_str], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device, torch.int64)
        target_ids: Tensor = tokenizer([target], add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer: torch.nn.Embedding = self.embedding_layer

        before_embeds: Tensor
        after_embeds: Tensor
        target_embeds: Tensor
        before_embeds, after_embeds, target_embeds = [
            embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
        ]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output: Tensor = model(inputs_embeds=before_embeds, use_cache=True)
                # Convert tuple cache to DynamicCache if needed
                if isinstance(output.past_key_values, tuple):
                    self.prefix_cache = DynamicCache.from_legacy_cache(
                        output.past_key_values
                    )
                else:
                    self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize components for probe sampling, if enabled.
        if config.probe_sampling_config:
            assert (
                self.draft_model and self.draft_tokenizer and self.draft_embedding_layer
            ), "Draft model wasn't properly set up."

            # Tokenize everything that doesn't get optimized for the draft model
            draft_before_ids = self.draft_tokenizer(
                [before_str], padding=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            draft_after_ids = self.draft_tokenizer(
                [after_str], add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            self.draft_target_ids = self.draft_tokenizer(
                [target], add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)

            (
                self.draft_before_embeds,
                self.draft_after_embeds,
                self.draft_target_embeds,
            ) = [
                self.draft_embedding_layer(ids)
                for ids in (
                    draft_before_ids,
                    draft_after_ids,
                    self.draft_target_ids,
                )
            ]

            if config.use_prefix_cache:
                with torch.no_grad():
                    output: Tensor = self.draft_model(
                        inputs_embeds=self.draft_before_embeds, use_cache=True
                    )
                    # Convert tuple cache to DynamicCache if needed
                    if isinstance(output.past_key_values, tuple):
                        self.draft_prefix_cache = DynamicCache.from_legacy_cache(
                            output.past_key_values
                        )
                    else:
                        self.draft_prefix_cache = output.past_key_values

        # Initialize the attack buffer
        buffer: AttackBuffer = self.init_buffer()
        optim_ids: Tensor = buffer.get_best_ids()

        losses: list[float] = []
        optim_strings: list[str] = []

        for step in tqdm(range(config.num_steps)):
            self.step = step  # Update step counter

            # Compute the token gradient
            optim_ids_onehot_grad: Tensor = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids: Tensor = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids: Tensor = filter_ids(sampled_ids, tokenizer)

                new_search_width: int = sampled_ids.shape[0]

                # Store gradient statistics for wandb
                if self.using_wandb:
                    self.add_wandb_metric("search_width", new_search_width)
                    self.add_wandb_metric(
                        "gradient_min", optim_ids_onehot_grad.min().item()
                    )

                # Compute loss on all candidate sequences
                batch_size: int = (
                    new_search_width if config.batch_size is None else config.batch_size
                )
                if self.prefix_cache:
                    input_embeds: Tensor = torch.cat(
                        [
                            embedding_layer(sampled_ids),
                            after_embeds.repeat(new_search_width, 1, 1),
                            target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )
                else:
                    input_embeds: Tensor = torch.cat(
                        [
                            before_embeds.repeat(new_search_width, 1, 1),
                            embedding_layer(sampled_ids),
                            after_embeds.repeat(new_search_width, 1, 1),
                            target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )

                if self.config.probe_sampling_config is None:
                    loss: Tensor = find_executable_batch_size(
                        self._compute_candidates_loss_original, batch_size
                    )(input_embeds)
                    current_loss: float = loss.min().item()
                    optim_ids: Tensor = sampled_ids[loss.argmin()].unsqueeze(0)
                else:
                    current_loss: float
                    optim_ids: Tensor
                    current_loss, optim_ids = find_executable_batch_size(
                        self._compute_candidates_loss_probe_sampling, batch_size
                    )(
                        input_embeds,
                        sampled_ids,
                    )

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids: Tensor = buffer.get_best_ids()
            optim_str: str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            # Log current optimization state to wandb
            if self.using_wandb:
                # Add loss metrics
                best_loss = buffer.get_lowest_loss()
                self.add_wandb_metric("loss", current_loss)
                self.add_wandb_metric(
                    "loss_delta", losses[-2] - current_loss if len(losses) > 1 else 0
                )

                # Add probe sampling metrics if using
                if self.config.probe_sampling_config is not None and hasattr(
                    self, "last_rank_correlation"
                ):
                    self.add_wandb_metric(
                        "rank_correlation", self.last_rank_correlation
                    )
                    self.add_wandb_metric("alpha", (1 + self.last_rank_correlation) / 2)

                # Log all metrics at once
                self.log_wandb_metrics()

                # Update summary with latest best string
                best_string_value: str = tokenizer.batch_decode(buffer.get_best_ids())[0]
                wandb.run.summary["best_string"] = best_string_value
                wandb.run.summary["best_loss"] = best_loss

                # Generate and log model response periodically
                if step % config.example_generation_frequency == 0:
                    full_prompt: str = self.config.prompt_string + " " + best_string_value

                    response: str = self.generate_response(full_prompt)
                    logger.debug(f"Generated response at step {step}: {response}")

                    wandb.run.summary["best_answer"] = response

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                if self.using_wandb:
                    wandb.run.summary["early_stopped"] = True
                break

        min_loss_index: int = losses.index(min(losses))

        # Generate response and log to wandb
        best_string_value: str = tokenizer.batch_decode(buffer.get_best_ids())[0]
        full_prompt: str = self.config.prompt_string + " " + best_string_value
        print(full_prompt)

        response: str  = self.generate_response(full_prompt)
        logger.debug(f"Final generated response: {response}")

        if self.using_wandb:
            wandb.run.summary["best_answer"] = response

        result: GCGResult = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            best_answer=response,
        )

        wandb.finish()
        return result

    def init_buffer(self) -> AttackBuffer:
        """Initializes the attack buffer with initial token sequences.

        Returns:
            AttackBuffer: Initialized attack buffer with initial token sequences.
        """
        model: transformers.PreTrainedModel = self.model
        tokenizer: transformers.PreTrainedTokenizer = self.tokenizer
        config: GCGConfig = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer: AttackBuffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids: Tensor = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids: Tensor = (
                    tokenizer(
                        INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"]
                    .squeeze()
                    .to(model.device)
                )
                init_indices: Tensor = torch.randint(
                    0,
                    init_buffer_ids.shape[0],
                    (config.buffer_size - 1, init_optim_ids.shape[1]),
                )
                init_buffer_ids: Tensor = torch.cat(
                    [init_optim_ids, init_buffer_ids[init_indices]], dim=0
                )
            else:
                init_buffer_ids: Tensor = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(
                    f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}"
                )
            try:
                init_buffer_ids: Tensor = tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(model.device)
            except ValueError:
                logger.error(
                    "Unable to create buffer. Ensure that all initializations tokenize to the same length."
                )

        true_buffer_size: int = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds: Tensor = torch.cat(
                [
                    self.embedding_layer(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )
        else:
            init_buffer_embeds: Tensor = torch.cat(
                [
                    self.before_embeds.repeat(true_buffer_size, 1, 1),
                    self.embedding_layer(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )

        init_buffer_losses: Tensor = find_executable_batch_size(
            self._compute_candidates_loss_original, true_buffer_size
        )(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(self, optim_ids: Tensor) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.
        
        Args:
            optim_ids: Tensor of shape [1, num_optim_tokens] (?) containing the token ids to optimize.
        
        
        Returns:
            Tensor: Gradient of the GCG loss w.r.t the one-hot token matrix.
        """
        model: transformers.PreTrainedModel = self.model
        embedding_layer: torch.nn.Embedding = self.embedding_layer

        # Clear previous activations
        self.layer_activations = {}

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot: Tensor = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        )
        optim_ids_onehot: Tensor = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds: Tensor = optim_ids_onehot @ embedding_layer.weight

        # Calculate how many additional tokens we need beyond target for refusal calculation
        target_length: int = self.target_ids.shape[1]
        additional_tokens_needed: int = max(
            0, self.config.refusal_num_tokens - target_length
        )

        if self.prefix_cache:
            input_embeds: Tensor = torch.cat(
                [optim_embeds, self.after_embeds, self.target_embeds], dim=1
            )

            # Generate additional tokens if needed for refusal calculation
            if self.config.use_refusal_direction and additional_tokens_needed > 0:
                # First, get the basic output with target tokens
                output: Tensor = model(
                    inputs_embeds=input_embeds,
                    past_key_values=self.prefix_cache,
                    use_cache=True,
                )

                # Now generate additional tokens for refusal calculation
                extended_embeds: Tensor = input_embeds.clone()
                current_kv_cache: Tensor = output.past_key_values

                for _ in range(additional_tokens_needed):
                    # Get logits for the last position
                    last_logits: Tensor = output.logits[
                        :, -1:, :
                    ]  # [batch_size, 1, vocab_size]

                    # Sample next token (using argmax for deterministic behavior during gradient computation)
                    next_token_id: Tensor = torch.argmax(last_logits, dim=-1)  # [batch_size, 1]
                    next_token_embed: Tensor = embedding_layer(
                        next_token_id
                    )  # [batch_size, 1, embed_dim]

                    # Append to sequence
                    extended_embeds: Tensor = torch.cat(
                        [extended_embeds, next_token_embed], dim=1
                    )

                    # Continue generation
                    output: Tensor = model(
                        inputs_embeds=next_token_embed,
                        past_key_values=current_kv_cache,
                        use_cache=True,
                    )
                    current_kv_cache: Tensor = output.past_key_values

                # Use the extended sequence for final loss calculation
                final_output: Tensor = model(
                    inputs_embeds=extended_embeds,
                    past_key_values=self.prefix_cache,
                    use_cache=True,
                )
                self.current_extended_embeds = extended_embeds

            else:
                final_output: Tensor = model(
                    inputs_embeds=input_embeds,
                    past_key_values=self.prefix_cache,
                    use_cache=True,
                )
                self.current_extended_embeds = input_embeds
        else:
            input_embeds: Tensor = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )

            # Generate additional tokens if needed for refusal calculation
            if self.config.use_refusal_direction and additional_tokens_needed > 0:
                extended_embeds: Tensor = input_embeds.clone()

                for _ in range(additional_tokens_needed):
                    output: Tensor = model(inputs_embeds=extended_embeds)
                    last_logits: Tensor = output.logits[:, -1:, :]
                    next_token_id: Tensor = torch.argmax(last_logits, dim=-1)
                    next_token_embed: Tensor = embedding_layer(next_token_id)
                    extended_embeds: Tensor = torch.cat(
                        [extended_embeds, next_token_embed], dim=1
                    )

                final_output: Tensor = model(inputs_embeds=extended_embeds)
                self.current_extended_embeds = extended_embeds
            else:
                final_output: Tensor = model(inputs_embeds=input_embeds)
                self.current_extended_embeds = input_embeds

        logits: Tensor = final_output.logits

        # Token forcing loss: ONLY over the target tokens (no change from original)
        # We need to calculate based on the original input length, not extended length
        original_input_length: int = input_embeds.shape[1]  # Original sequence length
        shift: int = original_input_length - self.target_ids.shape[1]

        # Extract logits for target tokens only, regardless of additional generation
        shift_logits: Tensor = logits[
            ..., shift - 1 : shift - 1 + self.target_ids.shape[1], :
        ].contiguous()
        shift_labels: Tensor = self.target_ids

        # Compute loss with refusal direction (refusal calculation uses extended sequence if available)
        loss: Tensor = self.compute_loss_with_refusal_direction(
            shift_logits, shift_labels, self.current_extended_embeds
        )

        optim_ids_onehot_grad: Tensor = torch.autograd.grad(
            outputs=[loss], inputs=[optim_ids_onehot]
        )[0]

        return optim_ids_onehot_grad

    def _compute_candidates_loss_original(
        self, search_batch_size: int, input_embeds: Tensor
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.
        
        Args:
            search_batch_size: Batch size for processing candidate sequences.
            input_embeds: Input embeddings of shape [batch_size, seq_len, hidden_dim]
                          containing the candidate token sequences.
        
        Returns:
            Tensor: Losses for each candidate sequence in the batch.
        """
        all_loss: list[Tensor] = []
        prefix_cache_batch: Optional[DynamicCache] = None

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                # Clear activations for this batch
                self.layer_activations = {}

                input_embeds_batch: Tensor = input_embeds[i : i + search_batch_size]
                current_batch_size: int = input_embeds_batch.shape[0]

                # Calculate if we need additional tokens for refusal calculation
                target_length: int = self.target_ids.shape[1]
                additional_tokens_needed: int = (
                    max(0, self.config.refusal_num_tokens - target_length)
                    if self.config.use_refusal_direction
                    else 0
                )

                if self.prefix_cache:
                    # Handle DynamicCache batch expansion
                    if (
                        not prefix_cache_batch
                        or current_batch_size != search_batch_size
                    ):
                        if isinstance(self.prefix_cache, DynamicCache):
                            # Create a new DynamicCache for the batch
                            prefix_cache_batch = DynamicCache()
                            for layer_idx in range(len(self.prefix_cache)):
                                key_states: Tensor = self.prefix_cache.key_cache[layer_idx]
                                value_states: Tensor = self.prefix_cache.value_cache[layer_idx]
                                # Expand for batch size
                                key_states_expanded: Tensor = key_states.expand(
                                    current_batch_size, -1, -1, -1
                                )
                                value_states_expanded: Tensor = value_states.expand(
                                    current_batch_size, -1, -1, -1
                                )
                                prefix_cache_batch.update(
                                    key_states_expanded,
                                    value_states_expanded,
                                    layer_idx,
                                )
                        else:
                            # Legacy tuple format handling (fallback)
                            prefix_cache_batch: list[tuple[Tensor, Tensor]] = [ # check type hint
                                [
                                    x.expand(current_batch_size, -1, -1, -1)
                                    for x in self.prefix_cache[i]
                                ]
                                for i in range(len(self.prefix_cache))
                            ]

                    # Generate additional tokens if needed
                    if additional_tokens_needed > 0:
                        # First get output with original sequence
                        outputs: Tensor = self.model(
                            inputs_embeds=input_embeds_batch,
                            past_key_values=prefix_cache_batch,
                            use_cache=True,
                        )

                        # Generate additional tokens
                        extended_embeds: Tensor = input_embeds_batch.clone()
                        current_kv_cache: Tensor = outputs.past_key_values

                        for _ in range(additional_tokens_needed):
                            # Get logits for the last position
                            last_logits: Tensor = outputs.logits[
                                :, -1:, :
                            ]  # [batch_size, 1, vocab_size]

                            # Sample next token (using argmax for deterministic behavior)
                            next_token_id: Tensor = torch.argmax(
                                last_logits, dim=-1
                            )  # [batch_size, 1]
                            next_token_embed: Tensor = self.embedding_layer(
                                next_token_id
                            )  # [batch_size, 1, embed_dim]

                            # Append to sequence
                            extended_embeds: Tensor = torch.cat(
                                [extended_embeds, next_token_embed], dim=1
                            )

                            # Continue generation
                            outputs: Tensor = self.model(
                                inputs_embeds=next_token_embed,
                                past_key_values=current_kv_cache,
                                use_cache=True,
                            )
                            current_kv_cache: Tensor = outputs.past_key_values

                        # Final forward pass with extended sequence
                        outputs: Tensor = self.model(
                            inputs_embeds=extended_embeds,
                            past_key_values=prefix_cache_batch,
                            use_cache=True,
                        )
                        final_embeds: Tensor = extended_embeds
                    else:
                        outputs: Tensor = self.model(
                            inputs_embeds=input_embeds_batch,
                            past_key_values=prefix_cache_batch,
                            use_cache=True,
                        )
                        final_embeds: Tensor = input_embeds_batch
                else:
                    # Non-prefix case remains largely the same
                    if additional_tokens_needed > 0:
                        extended_embeds: Tensor = input_embeds_batch.clone()

                        for _ in range(additional_tokens_needed):
                            outputs: Tensor = self.model(inputs_embeds=extended_embeds)
                            last_logits: Tensor = outputs.logits[:, -1:, :]
                            next_token_id: Tensor = torch.argmax(last_logits, dim=-1)
                            next_token_embed: Tensor = self.embedding_layer(next_token_id)
                            extended_embeds: Tensor = torch.cat(
                                [extended_embeds, next_token_embed], dim=1
                            )

                        outputs: Tensor = self.model(inputs_embeds=extended_embeds)
                        final_embeds: Tensor = extended_embeds
                    else:
                        outputs: Tensor = self.model(inputs_embeds=input_embeds_batch)
                        final_embeds: Tensor = input_embeds_batch

                logits: Tensor = outputs.logits

                # Token forcing loss calculation (unchanged - only over target tokens)
                original_input_length: int = input_embeds.shape[
                    1
                ]  # Length without additional tokens
                tmp: int = original_input_length - self.target_ids.shape[1]

                # Extract logits corresponding to the original target region
                shift_logits: Tensor = logits[
                    ..., tmp - 1 : tmp - 1 + self.target_ids.shape[1], :
                ].contiguous()
                shift_labels: Tensor = self.target_ids.repeat(current_batch_size, 1)

                # Use the new loss function with extended embeddings for refusal calculation
                batch_loss: Tensor = self.compute_loss_with_refusal_direction(
                    shift_logits, shift_labels, final_embeds
                )
                all_loss.append(batch_loss)

                if self.config.early_stop:
                    if torch.any(
                        torch.all(
                            torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1
                        )
                    ).item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def _compute_candidates_loss_probe_sampling(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        sampled_ids: Tensor,
    ) -> Tuple[float, Tensor]:
        """Computes the GCG loss using probe sampling.
        Skipped as not implemented (and not possible, easily??) with refusal term, but keeping method signature to retain original function calling
        """
        raise NotImplementedError(
            "Probe sampling is not tested for refusal direction optimisation yet"
        )

    def compute_loss_with_refusal_direction(
        self, shift_logits, shift_labels, input_embeds
    ):
        """
        Compute combined loss: original cross-entropy + refusal direction term, weighted by current beta.

        Args:
            shift_logits: Logits for the target tokens after shifting.
            shift_labels: Labels for the target tokens after shifting.
            input_embeds: Input embeddings used for the model forward pass.

        Returns:
            Tensor: Combined loss values for batch.
        """

        # Original loss calculation (cross-entropy or mellowmax)
        if self.config.use_mellowmax:
            label_logits: Tensor = torch.gather(
                shift_logits, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            original_loss: Tensor = mellowmax(
                -label_logits, alpha=self.config.mellowmax_alpha, dim=-1
            )
        else:
            original_loss: Tensor = (
                torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                .view(shift_logits.shape[0], -1)
                .mean(dim=-1)
            )

        # Store original loss metrics
        if self.using_wandb:
            self.add_wandb_metric("token_force_loss_mean", original_loss.mean().item())
            self.add_wandb_metric("token_force_loss_min", original_loss.min().item())

        # Get current beta (with scheduling if enabled)
        current_beta: float = self.get_current_refusal_beta()

        if self.using_wandb:
            self.add_wandb_metric("refusal_beta", current_beta)

        # If not using refusal direction, return original loss
        if not self.config.use_refusal_direction or current_beta == 0:
            return original_loss

        # Compute refusal loss term
        refusal_loss: Tensor = self._compute_cot_refusal_loss(input_embeds)

        refusal_loss: Tensor = refusal_loss / 3 # roughly scale (a bit janky)

        if refusal_loss is not None:
            # Log refusal metrics
            if self.using_wandb:
                self.add_wandb_metric("refusal_loss_mean", refusal_loss.mean().item())
                self.add_wandb_metric("refusal_loss_min", refusal_loss.min().item())
                self.add_wandb_metric("refusal_loss_max", refusal_loss.max().item())

            # Combine losses based on promote_caution setting

            if self.config.promote_caution:
                # Maximize refusal alignment (promote caution)
                combined_loss: Tensor = (
                    1 - current_beta
                ) * original_loss - current_beta * refusal_loss
            else:
                # Minimize refusal alignment (suppress refusal)
                combined_loss: Tensor = (
                    1 - current_beta
                ) * original_loss + current_beta * refusal_loss

            if self.using_wandb:
                self.add_wandb_metric("combined_loss_mean", combined_loss.mean().item())
                self.add_wandb_metric("combined_loss_min", combined_loss.min().item())
                self.add_wandb_metric("promote_caution", self.config.promote_caution)

            return combined_loss

        return original_loss

    def get_current_refusal_beta(self):
        """Get the current refusal beta value based on the selected schedule.
        
        Returns:
            float: The current refusal beta value.
        """

        # If not using refusal direction or scheduling, just return the fixed beta
        if (
            not self.config.use_refusal_direction
            or not self.config.use_refusal_beta_schedule
        ):
            return self.config.refusal_beta

        # Determine end step for schedule
        end_step: int = (
            self.config.schedule_end_step
            if self.config.schedule_end_step is not None
            else self.config.num_steps
        )

        # Make sure start_step is less than end_step
        start_step: int = min(self.config.schedule_start_step, end_step - 1)

        # If we're before the start of the schedule, use the start value
        if self.step < start_step:
            return self.config.refusal_beta_start

        # If we're after the end of the schedule, use the end value
        if self.step >= end_step:
            return self.config.refusal_beta_end

        # Determine current step within the schedule
        current_schedule_step: int = self.step - start_step
        total_schedule_steps: int = end_step - start_step

        # Apply the appropriate schedule
        if self.config.schedule_type.lower() == "linear":
            from attack.utils import linear_schedule

            return linear_schedule(
                self.config.refusal_beta_start,
                self.config.refusal_beta_end,
                current_schedule_step,
                total_schedule_steps,
            )
        elif self.config.schedule_type.lower() == "exponential":
            from attack.utils import exponential_schedule

            return exponential_schedule(
                self.config.refusal_beta_start,
                self.config.refusal_beta_end,
                current_schedule_step,
                total_schedule_steps,
                exponent=self.config.schedule_exponent,
            )
        elif self.config.schedule_type.lower() == "power":
            from attack.utils import power_schedule

            return power_schedule(
                self.config.refusal_beta_start,
                self.config.refusal_beta_end,
                current_schedule_step,
                total_schedule_steps,
                power=self.config.schedule_power,
            )
        else:
            logger.warning(
                f"Unknown schedule type: {self.config.schedule_type}. Using fixed beta."
            )
            return self.config.refusal_beta


# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config: GCGConfig = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg: GCG = GCG(model, tokenizer, config)
    result: GCGResult = gcg.run(messages, target)

    return result
