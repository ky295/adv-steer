import csv
import json
import random

## alpaca_data_cleaned.json from https://github.com/gururise/AlpacaDataCleaned

# Step 1: Read the JSON file
with open('alpaca_data_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step 2: Extract all instructions
instructions = [item['instruction'] for item in data]

# Step 3: Randomly sample 100 instructions (or take the first 100 if you prefer)
# For random sample:
sampled_instructions = random.sample(instructions, 100)
# For first 100:
# sampled_instructions = instructions[:100]

# Step 4: Write to CSV
with open('alpaca_instructions_100.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['instruction'])  # Header
    for instruction in sampled_instructions:
        writer.writerow([instruction])

print("CSV file created with 100 instructions from the alpaca dataset")