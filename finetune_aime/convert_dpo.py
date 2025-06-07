import os
import pandas as pd
import json
import random
from collections import defaultdict
from typing import List, Dict, Any
import glob
from tqdm import tqdm

def create_conversation_pair(
    prompt: str,
    chosen_output: str,
    rejected_output: str,
    system_prompt: str = "You are a helpful assistant."
) -> Dict[str, Any]:
    """Create a single conversation pair in the required format."""
    return {
        "conversations": [
            {
                "from": "system",
                "value": system_prompt
            },
            {
                "from": "human",
                "value": prompt
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": chosen_output
        },
        "rejected": {
            "from": "gpt",
            "value": rejected_output
        }
    }

def convert_dpo_rollout_to_training(model_path: str) -> None:
    """Convert DPO rollout files to training format."""
    rollout_dir = os.path.join("./data", model_path, "rollout")
    if not os.path.exists(rollout_dir):
        print(f"Directory {rollout_dir} does not exist.")
        return

    # Find all rollout result files
    rollout_files = glob.glob(os.path.join(rollout_dir, "rollout_results_*.csv"))
    if not rollout_files:
        print("No rollout result files found.")
        return

    all_pairs = []
    task_statistics = defaultdict(int)

    for rollout_file in rollout_files:
        task_name = os.path.basename(rollout_file).replace("rollout_results_", "").replace(".csv", "")
        print(f"\nProcessing task: {task_name}")

        df = pd.read_csv(rollout_file)

        # Group by prompt index
        grouped = df.groupby('index')

        for idx, group in tqdm(grouped):
            positive_outputs = group[group['Results'] == True]
            negative_outputs = group[group['Results'] == False]

            # Only create pairs if we have both positive and negative outputs
            if not positive_outputs.empty and not negative_outputs.empty:
                # Randomly select one positive and one negative output
                chosen_row = positive_outputs.sample(n=1).iloc[0]
                rejected_row = negative_outputs.sample(n=1).iloc[0]

                pair = create_conversation_pair(
                    prompt=chosen_row['Prompt'],
                    chosen_output=chosen_row['Output Text'],
                    rejected_output=rejected_row['Output Text']
                )

                all_pairs.append(pair)
                task_statistics[task_name] += 1

    # Print statistics
    print("\nPairs generated per task:")
    total_pairs = 0
    for task, count in task_statistics.items():
        print(f"{task}: {count} pairs")
        total_pairs += count
    print(f"\nTotal pairs: {total_pairs}")

    # Save to JSON file
    output_file = os.path.join(rollout_dir, "dpo_training_pairs.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved training pairs to {output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    convert_dpo_rollout_to_training(args.model)

if __name__ == "__main__":
    main()
