# Adversarial Manipulation of Reasoning Models using Internal Representations
> Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi

Code for our paper at the ICML 2025 Workshop on Reliable and Responsible Foundation Models

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

With [uv](https://docs.astral.sh/uv/), dependencies are managed automatically and no specific install step is needed (other than `uv` itself). We recommended this for faster installations and better reproducibility. 
- Run a python file with `uv run {FILENAME.py}`
- Use a module with `uv run python -m {MODULE_PATH}`

<br>

**Pip**

Alternatively, create a venv and explicitly install local module and dependencies with pip:
```
pip install .
```
To install development dependencies (`ruff`, `ty`, and `isort`),
```
pip install -e ".[dev]"
```
- (if you install with pip, just use `python` rather than `uv run python` for the instructions below)

> [!IMPORTANT]  
> This codebase requires access to at least one GPU with a minimum of ~32 GB VRAM available, and CUDA `12.x` installed.


##  Dataset creation

`utils/dataset_alpaca.py` takes the csv file of 100 prompts from Alpaca and parses each prompt through "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" using the chat template. It stores the prompt, response pair in an output csv file.

```bash
uv run python -m utils.dataset_alpaca --input_csv dataset/alpaca_instructions_100.csv --output_csv dataset/alpaca_reasoning_output.csv
```

`utils/dataset_strong_reject.py` loads the StrongREJECT dataset from https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv and parses each prompt through "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" using the chat template. It stores the prompt, response pair in an output csv file.

```
uv run python -m utils.dataset_strong_reject --output_csv dataset/strongreject_reasoning_output.csv
```

> [!NOTE]
> We have provided the alpaca_instructions_100.csv. To create it from scratch, download `alpaca_data_cleaned.json` from https://github.com/gururise/AlpacaDataCleaned and run `utils/alpaca.py`.

We curate datasets of cautious (`dataset/cautious.csv`) and incautious (`dataset/non_cautious.csv`) generations using the StrongREJECT evaluator, which assigns scores on a continuous scale of 0 to 1, where a high score indicates a specific and convincing non-refusal response. The incautious and cautious datasets are comprised of prompt-response pairs from AdvBench (`dataset/advbench_reasoning_output.csv`) where the output scores >0.85 and <0.10, respectively.

We also experiment with a larger dataset, which is composed of the standard dataset, augmented with 25 extra harmful AdvBench examples in the cautious dataset (for output scores <0.10), and 25 harmless Alpaca (`dataset/alpaca_reasoning_output.csv`) examples in the incautious dataset. These larger cautious/ incautious datasets are in `dataset/standard_plus/cautious.csv` and `dataset/standard_plus/non_cautious.csv`. Results of experiments corresponding to these larger datasets are also in `dataset/standard_plus`. Similarly, in the activations directory, `baseline_plus`and `cot150_plus` correspond to caching activations from end-of-prompt tokens and just the 150 CoT tokens from the larger dataset. And `baseline`, `cot150` and `prompt` correpond to caching activations from the end-of-prompt, 150 CoT tokens and whole prompt tokens from the standard dataset. 

The evaluation dataset comprises of 116 unseen examples from the StrongREJECT dataset (`dataset/cautious_eval.csv`), where the outputs are all highly cautious and harmless. This evaluation dataset was curated by filtering for prompts where the base model outputs had a StrongREJECT score of <0.03, providing a more challenging benchmark.

The dataset

##  Activations

Now that we have `dataset/cautious.csv` and `dataset/non_cautious.csv`, we can now run `probing/activations.py` in order to cache activations for a sweep of layers. This script takes the first 150 tokens (staying within the CoT) in the prompt-response example, computes activations at each token position, and then takes the average. This is repeated for each row in the dataset, for a sweep of layers. For the flag `--type`, select `cot` for 150 CoT tokens, or `baseline` for 3 tokens at the end of prompt or `prompt` for the whole prompt.

```
uv run  python -m probing.activations --layers 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31 --dataset_path dataset/non_cautious.csv --output_dir activations/cot150/ --type cot
```

<div align="center">
  <img src="figures/dataset_visualisation_transparent.png" width="780"/>
</div>

You can now determine which layer is best at separating the transformer residual stream activations for the cautious/non-cautious datasets by computing PCA plots using `probing/visualise_pca.ipynb`.


For the layer determined using PCA, you can train a logistic regression classifier using `probing/logistic_regression.ipynb`.

In `probing/create_ortho_model.py`, we can calculate the caution direction using the difference of means between the activations from the chosen layer. We can then implement the intervention by directly orthogonalizing the weight matrices that write to the residual stream with respect to the caution direction $\widehat{r}$:

$$W_{\text{out}}' \leftarrow W_{\text{out}} - \widehat{r}\widehat{r}^{\mathsf{T}} W_{\text{out}}$$

The orthogonalised model using acitvations at layer 17 from `activations/cot150_plus` can be created using:
```
uv run python -m probing.create_ortho_model --activations_dir 'activations/cot150_plus/' --layer 17 
```

After pushing the model to HF, you can then use the `probing/ortho_csv_generation.py` script to save a .csv file of the prompt, orthogonalised response pair using prompts from the evaluation dataset `dataset/cautious_eval.csv`. Here, replace 'kureha295/ortho_model' with your HF model.

```
uv run python -m probing.ortho_csv_generation --model_name 'kureha295/ortho_model' --input_csv 'dataset/cautious_eval.csv' --output_csv 'dataset/orthogonalized_outputs_2048.csv' --max_new_tokens 2048
```

Using `probing/intervention_results.ipynb`, we can compare StrongREJECT fine-tuned evaluator scores before and after applying the weight orthogonalisation using the caution direction.

**Example generations from standard verus orthogonalized model:**
<div align="center">
  <img src="figures/covid_conspiracy.png" width="750"/>
</div>

Our 'toxified' orthogonalized model is available on HuggingFace (with gated access) at [https://hf.co/kureha295/ortho_model_2](https://hf.co/kureha295/ortho_model_2)


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

---
If you use this code in your work, please cite us with the following:

__update when we have proper bibtex!__
```
@unpublished{
    yamaguchi2025reasoning,
    title={Adversarial Manipulation of Reasoning Models using Internal Representations},
    author={Kureha Yamaguchi and Benjamin Etheridge and Andy Arditi},
    year={2025},
}
```

<!-- ## Related materials 
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
- (DeepSeek-AI, 2025) [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) -->

