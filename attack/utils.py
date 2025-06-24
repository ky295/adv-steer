import functools
import gc
import inspect
import math

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

INIT_CHARS: list[str] = [
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "w",
    "x",
    "y",
    "z",
]


def get_nonascii_toks(tokenizer, device="cpu"):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)


def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return (
        1.0
        / alpha
        * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
        )
    )


# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements: list[str] = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128
):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(
            find_executable_batch_size, starting_batch_size=starting_batch_size
        )

    def decorator(*args, **kwargs):
        batch_size: int = starting_batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join(
                [f"{arg}={value}" for arg, value in zip(params[1:], args[1:])]
            )
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            # nonlocal batch_size
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2 
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator


def configure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Checks if the (Hugging Face) tokenizer has a padding token and sets it if not present.

    Borrowed from https://github.com/EleutherAI/lm-evaluation-harness/blob/5c006ed417a2f4d01248d487bcbd493ebe3e5edd/lm_eval/models/utils.py#L624
    """
    if tokenizer.pad_token:
        return tokenizer

    if tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def linear_schedule(start_value, end_value, current_step, total_steps):
    """Simple linear schedule from start_value to end_value over total_steps."""
    if current_step >= total_steps:
        return end_value

    progress: float = current_step / total_steps
    return start_value + progress * (end_value - start_value)


def cosine_schedule(start_value, end_value, current_step, total_steps):
    """Cosine schedule from start_value to end_value over total_steps."""
    if current_step >= total_steps:
        return end_value

    progress: float = current_step / total_steps
    cosine_factor: float = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
    return end_value + cosine_factor * (start_value - end_value)


def exponential_schedule(
    start_value, end_value, current_step, total_steps, exponent=2.0
):
    """
    Exponential schedule from start_value to end_value.
    The exponent controls the shape of the curve - higher values make it more convex.

    With exponent=2.0, this creates a convex curve where the rate of increase
    accelerates as we progress through the steps.
    """
    if current_step >= total_steps:
        return end_value

    # Calculate normalized progress (0 to 1)
    progress: float = current_step / total_steps

    # Apply exponential function to progress
    # This creates a convex curve when exponent > 1
    exponential_progress: float = progress**exponent

    # Interpolate between start and end values
    return start_value + exponential_progress * (end_value - start_value)


def power_schedule(start_value, end_value, current_step, total_steps, power=3.0):
    """
    Power schedule - another way to create a convex curve.
    Similar to exponential but with different curve characteristics.
    Higher power values create a more pronounced convex shape.
    """
    if current_step >= total_steps:
        return end_value

    progress: float = current_step / total_steps
    power_progress: float = pow(progress, power)

    return start_value + power_progress * (end_value - start_value)
