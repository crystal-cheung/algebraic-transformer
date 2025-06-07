from eval_benchmark import EvalModel
import csv
from utils import extract_answer_from_response
import os
import pandas as pd
import argparse
from tqdm import tqdm
from grading.grader import grade_answer


class EvalModelRollout(EvalModel):
    def __init__(self, 
                 model_path: str,
                 use_sglang: bool = True,
                 max_new_tokens: int = 4096,
                 temperature: float = 0.0,
                 dp_size: int = 4,
                 tp_size: int = 1,
                 rollout_size: int = 10,
                 task: str = "train"):
        super().__init__(model_path, use_sglang, max_new_tokens, temperature, dp_size, tp_size)
        self.rollout_size = rollout_size
        self.task = task
        
    def generate_rollout(self, debug: bool = False) -> None:
        # Define the data directory where the CSVs were saved
        data_dir = f"./data/{self.model_path}"
        predictions_csv = os.path.join(data_dir, "predictions.csv")
        raw_io_csv = os.path.join(data_dir, "raw_input_output.csv")

        if not os.path.exists(predictions_csv):
            print(f"File {predictions_csv} does not exist.")
            return
        if not os.path.exists(raw_io_csv):
            print(f"File {raw_io_csv} does not exist.")
            return

        # Load the CSV files
        df_predictions = pd.read_csv(predictions_csv)
        df_raw = pd.read_csv(raw_io_csv)

        # select rows with given task
        df_predictions = df_predictions[df_predictions['Task'] == self.task]
        df_raw = df_raw[df_raw['Task'] == self.task]
        if debug:
            df_predictions = df_predictions.head(5)
            df_raw = df_raw.head(5)
        
        # Exploration sampling parameters as specified
        explore_sampling_params = {
            "max_new_tokens": 4096,
            "temperature": 1.0,
            "n": self.rollout_size
        }

        # Create results directory if it doesn't exist
        save_dir = os.path.join(f'./data/{self.model_path}/rollout')
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare CSV file
        csv_filename = os.path.join(save_dir, f'rollout_results_{self.task}.csv')
        csv_header = ['Prompt', 'index', 'Output Number', 'Output Text', 'Output Answer', 'Expected Answer', 'Results']
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            all_positive = 0
            all_negative = 0
            paired = 0
            
            # For each cumulative prompt, generate 5 outputs and compare with the expected output
            for i, prompt in tqdm(enumerate(df_raw['Raw Input']), total=len(df_raw)):
                try:
                    if self.use_sglang:
                        responses = self.model.generate(self.template_prompt(prompt), sampling_params=explore_sampling_params)
                        outputs = [resp['text'] for resp in responses]
                    else:
                        exit("no implementation for non-SGLang model")

                    print("\nExploration outputs:")
                    
                    positive = False
                    negative = False
                    
                    for j, out in enumerate(outputs):
                        extracted_answer = extract_answer_from_response(out)
                        # Convert both extracted and expected answers to strings
                        if isinstance(extracted_answer, (int, float)):
                            extracted_answer = str(extracted_answer)
                        expected_answer = str(df_predictions.iloc[i]["Expected Output"])
                        matches_expected = grade_answer(extracted_answer, expected_answer)
                        
                        # Save to CSV
                        writer.writerow([
                            prompt,
                            i,
                            j,
                            out,
                            extracted_answer,
                            expected_answer,
                            matches_expected
                        ])
                        
                        if matches_expected:
                            positive = True
                        else:
                            negative = True
                            
                    if positive and negative:
                        paired += 1
                        print(f"Prompt {i} is paired")
                    elif positive:
                        all_positive += 1
                        print(f"Prompt {i} is all positive")
                    elif negative:
                        all_negative += 1
                        print(f"Prompt {i} is all negative")
                
                except Exception as e:
                    print(f"Error during generation for step {i}: {e}")
                    exit(1)
                    
        print(f"All positive: {all_positive}")
        print(f"All negative: {all_negative}")
        print(f"Paired: {paired}")
        print(f"Saved to {csv_filename}")



def main(args):
    try:
        # Load the model using our new EvalModelRollout class
        eval_model = EvalModelRollout(args.model, use_sglang=True, dp_size=args.dp_size, tp_size=args.tp_size, rollout_size=args.rollout_size, task=args.task)

        # Run the exploration method for one question (for debugging)
        eval_model.generate_rollout(debug=args.debug)

    finally:
        # Cleanup SGLang engine if present
        if 'eval_model' in locals() and hasattr(eval_model, 'model'):
            del eval_model.model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--rollout_size", type=int, default=16)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    main(args)
