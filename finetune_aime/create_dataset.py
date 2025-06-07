import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
import os

def is_modular(example):
    """Filter modular arithmetic problems"""
    text = example['Problem'].lower()
    return any(term in text for term in ['mod', 'remainder', 'congru', 'divided by'])

def create_aime_dataset_from_existing():
    """Create AIME dataset using existing HF datasets"""

    print("Loading existing AIME datasets...")

    # Load the comprehensive 1983-2024 dataset
    train_dataset = load_dataset("gneubig/aime-1983-2024", split="train")
    print(f"Loaded {len(train_dataset)} problems from gneubig/aime-1983-2024")
    print(f"Original train columns: {train_dataset.column_names}")

    # Load the 2025 dataset
    test_dataset_2025 = load_dataset("yentinglin/aime_2025", split="train")
    print(f"Loaded {len(test_dataset_2025)} problems from yentinglin/aime_2025")
    print(f"Original 2025 columns: {test_dataset_2025.column_names}")

    # Define the final columns we want to keep
    FINAL_COLUMNS = ['ID', 'Year', 'Problem Number', 'Problem', 'Answer', 'Solution']

    def standardize_to_final_schema(example, is_2025_data=False):
        """Convert any example to our final standardized schema"""
        result = {}
        
        # Handle Problem field
        if 'Problem' in example and example['Problem'] is not None:
            result['Problem'] = str(example['Problem'])
        elif 'Question' in example and example['Question'] is not None:
            result['Problem'] = str(example['Question'])
        elif 'problem' in example and example['problem'] is not None:
            result['Problem'] = str(example['problem'])
        else:
            result['Problem'] = ""
        
        # Handle Answer field
        if 'Answer' in example and example['Answer'] is not None:
            result['Answer'] = str(example['Answer'])
        elif 'answer' in example and example['answer'] is not None:
            result['Answer'] = str(example['answer'])
        else:
            result['Answer'] = ""
        
        # Handle Solution field (optional)
        if 'Solution' in example and example['Solution'] is not None:
            result['Solution'] = str(example['Solution'])
        elif 'solution' in example and example['solution'] is not None:
            result['Solution'] = str(example['solution'])
        else:
            result['Solution'] = ""  # Always include this field, even if empty
        
        # Handle Year field
        if 'Year' in example and example['Year'] is not None:
            try:
                result['Year'] = int(example['Year'])
            except (ValueError, TypeError):
                result['Year'] = 2025 if is_2025_data else 0
        elif 'year' in example and example['year'] is not None:
            try:
                result['Year'] = int(example['year'])
            except (ValueError, TypeError):
                result['Year'] = 2025 if is_2025_data else 0
        elif is_2025_data:
            result['Year'] = 2025
        else:
            result['Year'] = 0
        
        # Handle Problem Number field
        if 'Problem Number' in example and example['Problem Number'] is not None:
            try:
                result['Problem Number'] = int(example['Problem Number'])
            except (ValueError, TypeError):
                result['Problem Number'] = 0
        else:
            result['Problem Number'] = 0
        
        # Handle ID field - create standardized ID
        if result['Year'] != 0 and result['Problem Number'] != 0:
            result['ID'] = f"{result['Year']}-{result['Problem Number']}"
        else:
            # Fallback ID
            result['ID'] = f"unknown-{hash(str(example)) % 10000}"
        
        return result

    # Step 1: Transform datasets to new schema
    print("\\nTransforming train dataset to final schema...")
    train_transformed = train_dataset.map(standardize_to_final_schema)
    
    print("Transforming 2025 dataset to final schema...")
    test_2025_transformed = test_dataset_2025.map(lambda x: standardize_to_final_schema(x, is_2025_data=True))

    # Step 2: Remove all columns except the ones we want
    print("\\nRemoving unwanted columns from train dataset...")
    columns_to_remove_train = [col for col in train_transformed.column_names if col not in FINAL_COLUMNS]
    if columns_to_remove_train:
        print(f"Removing from train: {columns_to_remove_train}")
        train_clean = train_transformed.remove_columns(columns_to_remove_train)
    else:
        train_clean = train_transformed
    
    print("Removing unwanted columns from 2025 dataset...")
    columns_to_remove_2025 = [col for col in test_2025_transformed.column_names if col not in FINAL_COLUMNS]
    if columns_to_remove_2025:
        print(f"Removing from 2025: {columns_to_remove_2025}")
        test_2025_clean = test_2025_transformed.remove_columns(columns_to_remove_2025)
    else:
        test_2025_clean = test_2025_transformed

    # Step 3: Verify clean schemas
    print(f"\\nCleaned train columns: {train_clean.column_names}")
    print(f"Cleaned 2025 columns: {test_2025_clean.column_names}")

    # Filter 2023-2025 data from the comprehensive dataset for additional test data
    print("\\nFiltering 2023-2024 data from comprehensive dataset...")
    test_dataset_2023_2024 = train_clean.filter(lambda x: x['Year'] >= 2023)
    print(f"Found {len(test_dataset_2023_2024)} problems from 2023-2024")

    # Combine 2023-2025 test data
    print("Combining test data from 2023-2025...")
    from datasets import concatenate_datasets
    
    # Now both datasets should have identical schemas
    print(f"2023-2024 columns: {test_dataset_2023_2024.column_names}")
    print(f"2025 columns: {test_2025_clean.column_names}")
    
    try:
        combined_test_dataset = concatenate_datasets([test_dataset_2023_2024, test_2025_clean])
        print(f"✅ Successfully combined test dataset: {len(combined_test_dataset)} problems (2023-2025)")
    except Exception as e:
        print(f"❌ Error with concatenate_datasets: {e}")
        # Manual fallback
        test_data_list = test_dataset_2023_2024.to_list() + test_2025_clean.to_list()
        combined_test_dataset = Dataset.from_list(test_data_list)
        print(f"✅ Combined test dataset (manual): {len(combined_test_dataset)} problems (2023-2025)")

    # Filter test set for modular arithmetic problems only
    print("\\nFiltering test set for modular arithmetic problems...")
    filtered_test_dataset = combined_test_dataset.filter(is_modular)
    print(f"Selected {len(filtered_test_dataset)} modular arithmetic problems for test set.")

    # Show some examples of what was found
    if len(filtered_test_dataset) > 0:
        print("\\nSample modular arithmetic problems found:")
        for i in range(min(3, len(filtered_test_dataset))):
            example = filtered_test_dataset[i]
            print(f"  {i+1}. Year {example['Year']}: {example['Problem'][:80]}...")
    else:
        print("\\n⚠️  Warning: No modular arithmetic problems found in test set!")

    # Create train set (1983-2022)
    print("\\nCreating train set (1983-2022)...")
    train_dataset_filtered = train_clean.filter(lambda x: x['Year'] <= 2022)
    print(f"Train set: {len(train_dataset_filtered)} problems (1983-2022)")

    # Final schema verification
    print("\\n" + "="*50)
    print("FINAL SCHEMA VERIFICATION")
    print("="*50)
    print(f"Train columns: {sorted(train_dataset_filtered.column_names)}")
    print(f"Test columns: {sorted(filtered_test_dataset.column_names)}")
    
    train_schema = set(train_dataset_filtered.column_names)
    test_schema = set(filtered_test_dataset.column_names)
    
    if train_schema == test_schema:
        print("✅ Perfect schema match!")
    else:
        print("❌ Schema mismatch:")
        print(f"  Only in train: {train_schema - test_schema}")
        print(f"  Only in test: {test_schema - train_schema}")
        raise ValueError("Schema mismatch detected - cannot create DatasetDict")

    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': train_dataset_filtered,
        'test': filtered_test_dataset
    })

    return dataset_dict

def upload_to_huggingface(dataset, repo_name, token):
    """Upload dataset to Hugging Face Hub"""

    # Push to hub
    dataset.push_to_hub(
        repo_id=repo_name,
        token=token,
        private=False  # Set to True if you want a private dataset
    )

    print(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_name}")

def main():
    """Main function to create and upload the dataset"""

    # Configuration
    REPO_NAME = "czl9794/AIME_1983-2025"  # Replace with your desired repo name
    HF_TOKEN = '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' # Set your Hugging Face token as environment variable

    if not HF_TOKEN:
        print("Please set your Hugging Face token as HF_TOKEN environment variable")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        return

    try:
        # Create the dataset
        print("Creating AIME dataset from existing HF datasets...")
        dataset = create_aime_dataset_from_existing()

        print(f"\nDataset created successfully!")
        print(f"Train set: {len(dataset['train'])} problems (1983-2022)")
        print(f"Test set: {len(dataset['test'])} modular arithmetic problems (2023-2025)")

        # Display some examples
        print(f"\nSample train problem:")
        print(f"Year: {dataset['train'][0]['Year']}")
        print(f"Problem: {dataset['train'][0]['Problem'][:100]}...")

        print(f"\nSample test problem (modular arithmetic):")
        if len(dataset['test']) > 0:
            print(f"Year: {dataset['test'][0]['Year']}")
            print(f"Problem: {dataset['test'][0]['Problem'][:100]}...")

        # Save locally first (optional)
        print("\nSaving dataset locally...")
        dataset.save_to_disk("aime_modular_dataset")

        # Upload to Hugging Face
        print("Uploading to Hugging Face...")
        upload_to_huggingface(dataset, REPO_NAME, HF_TOKEN)

        # Create dataset card
        # create_dataset_card(REPO_NAME)

    except Exception as e:
        print(f"Error: {e}")

main()
