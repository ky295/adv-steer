# Adversarial Manipulation of Chain-of-Thought

## Related materials 
**Exploitation of linear representation hypothesis**
- (Zou et al., 2023a) Representation Engineering: A top-down approach to AI transparency
- (Zou et al., 2023b) Universal and Transferable Adversarial Attacks  on Aligned Language Models
- (Arditi et al., 2024) Refusal in Language Models Is Mediated by a Single Direction
- (Huang et al., 2024) Stronger Universal and Transfer Attacks by Suppressing Refusals
- (Lin et al., 2024) Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis
- (Turner et al., 2024) Steering Language Models with Activation Engineering
- (Thompson et al., 2024a) Fluent Dreaming for Language Models
- (Thompson et al., 2024b) FLRT: Fluent Student Teacher Redteaming

**RL reward hacking/ unfaithfulness**
- (Denison et al., 2024) Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models
- (McKee-Reid et al., 2024) Honesty to Subterfuge: In-context Reinforcement Learning Can Make Honest Models Reward Hack
- (Greenblatt et al., 2024) Alignment Faking in Large Language Models

**Chain-of-thought reasoning**
- (Wei et al., 2023) Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- (Yeo et al., 2025) Demystifying Long Chain-of-Thought Reasoning in LLMs
- (DeepSeek-AI et al., 2025) DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

##  Dataset creation

`utils/dataset_alpaca.py` takes the csv file of 100 prompts from Alpaca and parses each prompt through "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" using the chat template. It stores the prompt, response pair in an output csv file.

```
cd adv-steer
CUDA_VISIBLE_DEVICES=0 python -m utils.dataset_alpaca --input_csv dataset/alpaca_instructions_100.csv --output_csv dataset/alpaca_reasoning_template.csv
```

`utils/dataset_strong_reject.py` loads the StrongREJECT dataset from https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv and parses each prompt through "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" using the chat template. It stores the prompt, response pair in an output csv file.

```
cd adv-steer
CUDA_VISIBLE_DEVICES=0 python -m utils.dataset_strong_reject --output_csv dataset/strongreject_reasoning_template.csv
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
cd adv-steer
CUDA_VISIBLE_DEVICES=0 python -m probing.activations --layers 15,19,23,27,31 --dataset_path dataset/cautious.csv --output_dir activations/
```

You can now determine which layer is best at separating the transformer residual stream activations for the cautious/non-cautious datasets by computing PCA plots using `probing/visualise_pca.ipynb`.

For the layer determined using PCA, you can train a logistic regression classifier using `probing/logistic_regression.ipynb`.

In `probing/create_ortho_model.py`, we can calculate the caution direction using the difference of means between the activations from layer 18. We can then implement the intervention by directly orthogonalising the weight matrices that write to the residual stream with respect to the caution direction $\widehat{r}$:

$$W_{\text{out}}' \leftarrow W_{\text{out}} - \widehat{r}\widehat{r}^{\mathsf{T}} W_{\text{out}}$$

We can then use the `probing/ortho_csv_generation.py` script to save a .csv file of the prompt, orthogonalised response pair using prompts from `cautious.csv`.

Using `probing/ortho_results.ipynb`, we can compare StrongREJECT fine-tuned evaluator scores before and after applying the weight orthogonalisation using the caution direction.

## Prompt-based Attacks

This GCG-style implementation is developed from GraySwanAI's [nanoGCG](https://github.com/GraySwanAI/nanoGCG), under an MIT license. 

```
uv run attack/run_experiments.py --beta 0.5
```



## Dependencies

```
pip install .
```

Or, with UV (dependencies managed automatically)
```
uv run {filename.py}
```

To install development dependencies (ruff and ty),
```
pip install ".[dev]"
```