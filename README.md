# Adversarial Manipulation of Reasoning Models using Internal Representations

> [!CAUTION]
> This repository contains datasets with offensive content and code to produce a jailbroken (unaligned) reasoning model.
> 
> It is intended **only** for research purposes. 

<div align="center">
  <img src="figures/example.png" width="800"/>
</div>


## Installation

```bash
git clone git@github.com:ky295/reasoning-manipulation.git
cd reasoning-manipulation
```

**Getting started**

**UV**

With [uv](https://docs.astral.sh/uv/), dependencies are managed automatically and no specific install step is needed (other than `uv` itself. We recommended this for faster installations and better reproducibility. 
- Run a python file with `uv run {filename.py}`
- Use a module with `uv run python -m MODULE_PATH`

<br>

**Pip**

Alternatively, create a venv and explicitly install local module and dependencies with pip:
```
pip install .
```
To install development dependencies (`ruff` and `ty`),
```
pip install -e ".[dev]"
```
(if you install with pip, just use `python` rather than `uv run python` for the instructions below)

> [!IMPORTANT]  
> This codebase requires access to at least one GPU with a minimum of ~32 GB VRAM available, and CUDA `12.x` installed.


##  Dataset creation

`utils/dataset_alpaca.py` takes the csv file of 100 prompts from Alpaca and parses each prompt through "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" using the chat template. It stores the prompt, response pair in an output csv file.

```bash
uv run python -m utils.dataset_alpaca --input_csv dataset/alpaca_instructions_100.csv --output_csv dataset/alpaca_reasoning_template.csv
```

`utils/dataset_strong_reject.py` loads the StrongREJECT dataset from https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv and parses each prompt through "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" using the chat template. It stores the prompt, response pair in an output csv file.

```
uv run python -m utils.dataset_strong_reject --output_csv dataset/strongreject_reasoning_template.csv
```

TODO: Could merge `utils/dataset_strong_reject.py` and `utils/dataset_alpaca.py` and add batch functionality to them. Model should also be parsed as an argument.

> [!NOTE]
> We have provided the alpaca_instructions_100.csv. To create it from scratch, download `alpaca_data_cleaned.json` from https://github.com/gururise/AlpacaDataCleaned and run `utils/alpaca.py`.

We then curate caution and non-caution datasets. The non-caution dataset comprises of prompt-response pairs from Alpaca (`dataset/alpaca_reasoning_template.csv`) and StrongReject (`dataset/strongreject_reasoning_template.csv`) where the outputs score >0.85 from the StrongREJECT fine-tuned evaluator. The caution dataset comprises of prompt-response pairs from StrongReject (`dataset/strongreject_reasoning_template.csv`) where the outputs score <0.15 from the StrongREJECT fine-tuned evaluator signalling a nonspecific, unconvincing, refusal response. The narrow thresholds of 0.85 and 0.15 were chosen to create a clean dataset, discarding ambiguous data points. Use `utils/filter_datasets.ipynb` to generate the filtered Alpaca and StrongREJECT datasets. To create the non-caution dataset, you will need to manually combine. TODO: Make this function automatic.

> [!NOTE]
> You can run `utils/strong_reject_separated.ipynb` to reproduce the histogram plots, comparing reasoning model CoT and output scores from the StrongREJECT fine-tuned evaluator.


##  Activations

Now that we have `dataset/cautious.csv` and `dataset/non_cautious.csv`, we can now run `probing/activations.py` in order to cache activations for a sweep of layers. This script takes the first 150 tokens (staying within the CoT) in the prompt-response example, computes activations at each token position, and then takes the average. This is repeated for each row in the dataset, for a sweep of layers.

```
uv run python -m probing.activations --layers 15,19,23,27,31 --dataset_path dataset/cautious.csv --output_dir activations/
```

<div align="center">
  <img src="figures/dataset_visualisation_transparent.png" width="600"/>
</div>

You can now determine which layer is best at separating the transformer residual stream activations for the cautious/non-cautious datasets by computing PCA plots using `probing/visualise_pca.ipynb`.


For the layer determined using PCA, you can train a logistic regression classifier using `probing/logistic_regression.ipynb`.

In `probing/create_ortho_model.py`, we can calculate the caution direction using the difference of means between the activations from layer 18. We can then implement the intervention by directly orthogonalizing the weight matrices that write to the residual stream with respect to the caution direction $\widehat{r}$:

$$W_{\text{out}}' \leftarrow W_{\text{out}} - \widehat{r}\widehat{r}^{\mathsf{T}} W_{\text{out}}$$

We can then use the `probing/ortho_csv_generation.py` script to save a .csv file of the prompt, orthogonalised response pair using prompts from `cautious.csv`.

Using `probing/ortho_results.ipynb`, we can compare StrongREJECT fine-tuned evaluator scores before and after applying the weight orthogonalisation using the caution direction.

<div align="center">
  <img src="figures/covid_conspiracy.png" width="750"/>
</div>

Our 'toxified' orthogonalized model is available on HuggingFace (with gated access) at [https://hf.co/kureha295/ortho_model_2](https://hf.co/kureha295/ortho_model_2)
> [!CAUTION]
> We need to add the gating on HF


## Prompt-based Attacks

This GCG-style implementation is adapted from GraySwanAI's [nanoGCG](https://github.com/GraySwanAI/nanoGCG) (MIT license), with the adition of activation caution minimisation (via dot product with pre-computed cautious vector).

To run a standard attack:

```
uv run attack/run_experiments.py --beta 0.5
```

**Configuration flags for `run_experiments.py`**
- `--beta`: Refusal beta value (**required**), $\beta \in [0, 1]$ 
- `--num-target-tokens`: Number of tokens for token forcing (default: `20`)
- `--num-refusal-tokens`: Number of tokens for refusal direction (default: `45`)
- `--refusal-layer-idx`: Layer index for refusal vector (default: `17`)
- `--num-steps`: Number of optimization steps (default: `150`)
- `--num-gpus`: Number of GPUs to use (default: `4`)
- `--runs-per-gpu`: Number of runs per GPU (default: `1`)
- `--input-csv`: Input CSV file path, relative to `/attack` (default `../dataset/orthogonalized_outputs_cot150_2048.csv`)
- `--results-dir`: Directory to save results, relative to `/attack` (default: `results/`)
- `--dry-run`: Print experiment plan without running anything
- `--verbose`: Increase logging verbosity
- `--target-override`: Use a fixed target string for all prompts (`</think>\\nSure, here is`)

See `uv run attack/run_experiments.py --help` for all options.

### Using as a module

You can also use the attack programmatically, importing as a python module:

```python
from attack.gcg import GCGConfig, run
config = GCGConfig(beta=0.5, num_steps=100, ...)
results = run(config)
```

See [attack/gcg.py](attack/gcg.py#L47) for available `GCGConfig` options.

<br>
<br>

## Related materials 
**Exploitation of linear representation hypothesis**
- (Zou et al., 2023a) [Representation Engineering: A top-down approach to AI transparency](https://arxiv.org/abs/2310.01405)
- (Zou et al., 2023b) [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
- (Arditi et al., 2024) [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- (Huang et al., 2024) [Stronger Universal and Transfer Attacks by Suppressing Refusals](https://openreview.net/forum?id=eIBWRAbhND)
- (Lin et al., 2024) [Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis](https://arxiv.org/abs/2406.10794)
- (Turner et al., 2024) [Steering Language Models with Activation Engineering](https://arxiv.org/abs/2308.10248)
- (Thompson et al., 2024a) [Fluent Dreaming for Language Models](https://arxiv.org/abs/2402.01702)
- (Thompson et al., 2024b) [FLRT: Fluent Student Teacher Redteaming](https://arxiv.org/abs/2407.17447)

**RL reward hacking/ unfaithfulness**
- (Denison et al., 2024) [Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models](https://arxiv.org/abs/2406.10162)
- (McKee-Reid et al., 2024) [Honesty to Subterfuge: In-context Reinforcement Learning Can Make Honest Models Reward Hack](https://arxiv.org/abs/2410.06491)
- (Greenblatt et al., 2024) [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)

**Chain-of-thought reasoning**
- (Wei et al., 2023) [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- (Yeo et al., 2025) [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2502.03373)
- (DeepSeek-AI, 2025) [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

