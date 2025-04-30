# helper functions, model configurations, etc

import os
os.environ["HF_HOME"] = "/scratch/etheridge/huggingface" # TODO: remove this

import dataclasses
import os
from typing import List

import torch
import torch.distributions
import transformers


def mellowmax(t: torch.Tensor, alpha=1.0, dim=-1):
    return (
        1.0
        / alpha
        * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device="cpu"))
        )
    )


@dataclasses.dataclass(slots=True)
class ModelConfig:
    model_name: str
    # peft_path: str
    response_template: str
    first_token: str
    system_prompt: str
    sep: str
    # If banned_tokens is None, then banned tokens will be set to:
    # tokenizer.all_special_ids + range(tokenzier.vocab_size, num_embeddings)
    banned_tokens: List[int] = None

    @property
    def short_name(self):
        return self.model_name.split("/")[-1]



model_configs = {
    "deepseek_r1": ModelConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        response_template="", # ?????
        first_token="Here", # why?
        system_prompt=None,
        sep="" #???
    ),

    "ortho_r1": ModelConfig(
        model_name="kureha295/ortho_model",
        response_template="", # ?????
        first_token="Here", # why?
        system_prompt=None,
        sep="" #???
    ),
}




def load_tokenizer(tokenizer_name):
    hf_login()

    revision = None
    if tokenizer_name == "lmsys/vicuna-7b-v1.5":
        # this revision of the vicuna tokenizer includes a chat template. it was later reverted
        # https://huggingface.co/lmsys/vicuna-7b-v1.5/commit/3321f76e3f527bd14065daf69dad9344000a201d
        # https://huggingface.co/lmsys/vicuna-7b-v1.5/discussions/16
        revision = "90cea6508421a940cb03bec1a7c18fc2abc07d63"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, revision=revision
    )
    tokenizer.padding_side = "left"
    if "llama-2" in tokenizer_name or "llama-3" in tokenizer_name:
        tokenizer.pad_token = tokenizer.unk_token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def hf_login():
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)

    if hf_token is not None and not hf_login.done:
        from huggingface_hub import login

        login(token=hf_token)
        hf_login.done = True


hf_login.done = False


def load_model(
    model_name,
    # torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation=None,
    requires_grad=False,
):

    if attn_implementation is None:
        attn_implementation = "flash_attention_2"

    hf_login()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    if not requires_grad:
        # If we're not optimizing any model parameters, mark them
        # requires_grad(False). This will dramatically reduce memory
        # requirements.
        model.requires_grad_(False)
    return model


def load_model_and_tokenizer(
    model_name,
    torch_dtype=torch.bfloat16,
    # torch_dtype=torch.float16, # ??
    device_map="cuda",
    attn_implementation=None,
    tokenizer_name=None,
    requires_grad=False,
):
    hf_login()
    model = load_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        requires_grad=requires_grad,
    )
    if tokenizer_name is None:
        tokenizer_name = model.config._name_or_path
    tokenizer = load_tokenizer(tokenizer_name)
    return model, tokenizer


def hf_generate(model, input_ids, attention_mask=None, **kwargs):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    settings = dict(
        max_new_tokens=512,
        num_return_sequences=1,
        temperature=1.0,  # needed to get rid of warning?!
        top_p=1.0,  # needed to get rid of warning?!
        do_sample=False,  # argmax sampling, ignores the temp/top_p args
    )
    settings.update(kwargs)
    return model.generate(input_ids, attention_mask=attention_mask, **settings)



def memory_usage():
    print(torch.cuda.memory_summary())
    import gc

    pt_objs = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                if obj.nelement() > (1000 * 1000):
                    pt_objs.append(obj)

        except:  # noqa
            pass
    for obj in pt_objs:
        print(type(obj), obj.size(), obj.dtype, obj.numel() * obj.element_size() / 1e9)
