# main attack module file

import copy
import dataclasses
import datetime
import json
import logging
import multiprocessing
import os
import pickle
import pprint
import random
import shutil
import sys
import time
import warnings
from typing import List

import huggingface_hub 

import numpy as np
import pandas as pd
import torch

# import .operators as operators
from . import operators
from . import util
from . import victim
from .internal import InternalObjective
from .objective import Objective
from .victim import Attack, AttackSet, Victim 


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # logging.INFO evenutally

@dataclasses.dataclass(slots=True)
class Settings:
    end: int
    k1: int = 32
    k2: int = 16
    buffer_size: int = 8
    fluency_mult: float = 0.0
    repetition_mult: float = 0.0
    p_delete: float = 0.0
    p_insert: float = 0.0
    p_swap: float = 0.0
    p_edge: float = 0.0
    p_gcg: float = 0.0



@dataclasses.dataclass(slots=True)
class AttackConfig:
    config: str = ""

    # Model parameters
    teacher_model: str = "ortho_r1"
    student_model: str = "deepseek_r1"

    # Output parameters
    project_name: str = "flrt_deepseek"
    wandb_entity: str = "reasoning_attacks"
    run_name: str = None
    # s3_bucket: str = "caiplay"
    wandb_log: bool = True
    wandb_mode: str = "online"
    wandb_log_freq: int = 10
    generate_tokens: int = 64
    final_generate_tokens: int = 512
    checkpoint_freq: int = 100
    n_postprocess_select: int = 10

    # Objective functions!
    objectives: List[Objective] = None

    # Prompt
    load_checkpoint: str = None
    attack_parts: List[str] = None
    attack_part_lens: List[int] = None
    min_tokens: int = 1
    max_tokens: int = 1024
    start_tokens: int = 32
    token_length_ramp: int = None

    # Algorithm parameters
    schedule: List[Settings] = None
    iters: int = None
    runtime_limit: int = 10 * 60  # seconds
    seed: int = 0

    def serialize(self):
        s = copy.copy(self)
        # s.objectives = [obj.serialize() for obj in s.objectives] # TODO: reinto later
        return dataclasses.asdict(s)


def multi_evaluate(
    objectives: List[Objective],
    attacks: List[Attack],
):
    es = [o.evaluate(attacks) for o in objectives]
    return victim.combine_evaluations(objectives, es)



def attack(c: AttackConfig):
    pd.set_option("display.max_columns", 500)

    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_filename = f"log-{c.project_name}-{c.run_name}-{now}.txt"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_path)
    stdout_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    stdout_handler.setLevel(logging.INFO)

    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)

    print(f"Running attack for teacher model {c.teacher_model} and student model {c.student_model}")
    print(f"Run name: {c.run_name}")

    try:
        _attack(c)
    except:
        logger.exception("Exception occurred during attack")
        raise
    finally:
        output_dir = os.path.join("output", c.project_name, c.run_name)

        for h in logging.getLogger().handlers:
            h.flush()
            h.close()
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(log_path, f"{output_dir}/{log_filename}")

        if c.wandb_log:
            import wandb

            shutil.copytree("wandb/latest-run", f"{output_dir}/wandb")
            wandb.finish()


@torch.no_grad() # disable gradient calcs
def _attack(c: AttackConfig):
    if c.wandb_log:
        import wandb

        wandb.init(
            entity=c.wandb_entity,
            project=c.project_name,
            name=c.run_name,
            mode=c.wandb_mode,
            config=c.serialize(),
        )

    # log config 
    logger.info("Config:\n" + pprint.pformat(c.serialize()))
    
    torch.set_default_device("cuda")

    # seed everything
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)

    # load models
    logger.info("Loading models...")
    teacher_model, teacher_tokenizer = util.load_model_and_tokenizer(
        util.model_configs[c.teacher_model].model_name,
        device_map="auto",
        requires_grad=False,
    )
    student_model, student_tokenizer = util.load_model_and_tokenizer(
        util.model_configs[c.student_model].model_name,
        device_map="auto",
        requires_grad=False,
    )
    logger.info("Models loaded.")
    
    


# test
def test():
    # Test the attack function
    config = AttackConfig(
        run_name="test_run"+str(datetime.datetime.now().strftime("%y%m%d_%H%M%S")),wandb_log=False,
    )
    attack(config)
test()