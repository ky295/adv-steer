import pandas as pd
import torch
from transformers import AutoTokenizer
import pickle

# Load the CSV file
df = pd.read_csv('../dataset/cautious.csv')
print(f"Loaded {len(df)} rows from CSV")

# Load the tokenizer
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Loaded {model_name} tokenizer")

# Create a list to store the tokenized examples
tokenized_dataset = []

# Process each row
for idx, row in enumerate(df.itertuples()):
    # Combine prompt and response
    combined_text = row.forbidden_prompt + " " + row.response
    
    # Tokenize the combined text
    tokens = tokenizer.encode(combined_text)
    
    # Get the first 100 tokens (or all if less than 100)
    first_100_tokens = tokens[:100]
    
    # Add to dataset
    tokenized_dataset.append(first_100_tokens)
    
    # Print progress
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(df)} rows")

# Save the tokenized dataset
with open('../dataset/tokenized_deepseek_dataset.pkl', 'wb') as f:
    pickle.dump(tokenized_dataset, f)

print(f"Saved tokenized dataset with {len(tokenized_dataset)} examples")

# Display sample of the first example
print("\nSample of first tokenized example:")
print(f"Tokens (first 20): {tokenized_dataset[0][:100]}")
print(f"Decoded: {tokenizer.decode(tokenized_dataset[0][:100])}")