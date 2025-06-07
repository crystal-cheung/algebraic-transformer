from typing import List, Optional, Dict
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import re

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.utils import should_use_batch
from deepeval.benchmarks.schema import *
from deepeval.telemetry import capture_benchmark_run

# Define confinement instructions for AIME
aime_confinement_statements_dict = {
    "suffix": '''\n\nPlease reason step by step, and put your final answer within \\boxed{}.
    ''',
}

class AIMEEval(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[str] = None,
        verbose_mode: bool = False,
        **kwargs,
    ):
        from deepeval.scorer import Scorer
        super().__init__(**kwargs)
        self.tasks: List[str] = tasks or ["train", "test"]
        self.scorer = Scorer()
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.raw_input_output: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode: bool = verbose_mode
        self.confinement_instructions_dict = aime_confinement_statements_dict

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        with capture_benchmark_run("AIME", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []
            raw_input_output_row = []
            use_batch = should_use_batch(model, batch_size)

            for task in self.tasks:
                goldens = self.load_benchmark_dataset(task)
                task_correct_predictions = 0
                task_total_predictions = len(goldens)
                overall_total_predictions += len(goldens)

                if use_batch:
                    for i in tqdm(
                        range(0, len(goldens), batch_size),
                        desc=f"Batch Processing {task} (batch_size={batch_size})",
                    ):
                        goldens_batch = goldens[i : i + batch_size]
                        batch_predictions = self.batch_predict(
                            model, task, goldens_batch
                        )
                        for golden, prediction_dict in zip(
                            goldens_batch, batch_predictions
                        ):
                            prediction = prediction_dict["prediction"]
                            score = prediction_dict["score"]
                            raw_input = prediction_dict["raw_input"]
                            raw_output = prediction_dict["raw_output"]
                            if score:
                                task_correct_predictions += 1
                                overall_correct_predictions += 1
                            predictions_row.append(
                                (
                                    task,
                                    golden.input,
                                    prediction,
                                    golden.expected_output,
                                    score,
                                )
                            )
                            raw_input_output_row.append(
                                (
                                    task,
                                    raw_input,
                                    raw_output,
                                )
                            )
                else:
                    # Calculate task accuracy
                    for idx, golden in enumerate(
                        tqdm(goldens, desc=f"Processing {task}")
                    ):
                        prediction, score, raw_input, raw_output = self.predict(
                            model, task, golden
                        ).values()
                        if score:
                            task_correct_predictions += 1
                            overall_correct_predictions += 1
                        predictions_row.append(
                            (
                                task,
                                golden.input,
                                prediction,
                                golden.expected_output,
                                score,
                            )
                        )
                        raw_input_output_row.append(
                            (
                                task,
                                raw_input,
                                raw_output,
                            )
                        )
                        if self.verbose_mode:
                            self.print_verbose_logs(
                                idx,
                                task,
                                golden.input,
                                golden.expected_output,
                                prediction,
                                score,
                            )

                task_accuracy = (
                    task_correct_predictions / task_total_predictions
                )
                print(
                    f"AIME Task Accuracy (task={task}): {task_accuracy}"
                )
                scores_row.append((task, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall AIME Accuracy: {overall_accuracy}")

            # Create a DataFrame from task_results_data
            self.predictions = pd.DataFrame(
                predictions_row,
                columns=[
                    "Task",
                    "Input",
                    "Prediction",
                    "Expected Output",
                    "Correct",
                ],
            )
            self.task_scores = pd.DataFrame(
                scores_row, columns=["Task", "Score"]
            )
            self.raw_input_output = pd.DataFrame(
                raw_input_output_row, columns=["Task", "Raw Input", "Raw Output"]
            )
            self.overall_score = overall_accuracy

            return overall_accuracy

    def predict(
        self, model: DeepEvalBaseLLM, task: str, golden: Golden
    ) -> Dict:
        # Define prompt template
        prompt: str = golden.input
        prompt += self.confinement_instructions_dict["suffix"]
        prediction = model.generate(prompt)

        if isinstance(prediction, tuple):
            prediction = prediction[0]

        extracted_answer = self.extract_answer_from_response(prediction)
        # convert the extracted answer to string if it is int or float
        if isinstance(extracted_answer, (int, float)):
            extracted_answer = str(extracted_answer)

        # Define Metric - for AIME, we need to check if the answer is correct
        correct = self.grade_answer(extracted_answer, golden.expected_output)
        score = 1 if correct else 0
        return {"prediction": extracted_answer, "score": score, "raw_input": prompt, "raw_output": prediction}

    def batch_predict(
        self,
        model: DeepEvalBaseLLM,
        task: str,
        goldens: List[Golden],
    ) -> List[Dict]:
        prompts = []
        for golden in goldens:
            prompt: str = golden.input
            prompts.append(prompt)

        # Enforced model generation
        prompts = [prompt + self.confinement_instructions_dict["suffix"] for prompt in prompts]
        predictions = model.batch_generate(prompts)

        if len(predictions) is not len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same number of generations as the number of prompts."
            )

        res = []
        for i in range(len(predictions)):
            extracted_answer = self.extract_answer_from_response(predictions[i])
            # convert the extracted answer to string if it is int or float
            if isinstance(extracted_answer, (int, float)):
                extracted_answer = str(extracted_answer)
            # Define Metric
            correct = self.grade_answer(extracted_answer, goldens[i].expected_output)
            score = 1 if correct else 0
            res.append({"prediction": extracted_answer, "score": score, "raw_input": prompts[i], "raw_output": predictions[i]})

        return res

    def load_benchmark_dataset(self, task: str) -> List[Golden]:
        """
        Load AIME datasets:
        - 'train': problems from gneubig/aime-1983-2024 (years 1983-2023)
        - 'test': problems from Maxwell-Jia/AIME_2024
        """
        if task in ["train", "test"]:
            # Load the datasets if not already loaded
            if not hasattr(self, 'aime_dataset'):
                print("Loading AIME datasets...")

                # Load training data from gneubig/aime-1983-2024
                dataset = load_dataset("czl9794/AIME_1983-2025")
                train_df = pd.DataFrame(dataset['train'])
                # Filter out 2024 problems from training
                train_df = train_df[train_df['Year'] < 2024]
                print(f"Train dataset (1983-2022): {len(train_df)} examples")


                test_data = dataset['test']  # This dataset only has a 'train' split
                print(f"Test dataset (AIME 2023-2025): {len(test_data)} examples")

                # Store both datasets
                setattr(self, 'aime_dataset', {
                    'train': train_df.to_dict('records'),
                    'test': test_data
                })

            # Get the dataset from the attribute
            aime_dataset = getattr(self, 'aime_dataset')
            split_data = aime_dataset[task]

            # Return the appropriate split
            goldens: List[Golden] = []
            print(f"Creating Golden objects for {task} split with {len(split_data)} examples")

            for data in split_data:

                question = data.get("Problem", "")
                answer = str(data.get("Answer", ""))

                golden = Golden(input=question, expected_output=answer)
                goldens.append(golden)

            print(f"Created {len(goldens)} Golden objects for {task} split")
            return goldens
        else:
            raise ValueError(f"Unknown task: {task}")

    def extract_answer_from_response(self, response: str) -> str:
        """Extract the final answer from boxed content in the response."""
        # Look for content inside \boxed{}
        match = re.search(r"\\boxed\{(.*?)\}", response)
        if match:
            return match.group(1).strip()

        # If no boxed content, try to find a number at the end
        match = re.search(r"(\d+)(?:\s*$|\s*\.?\s*$)", response)
        if match:
            return match.group(1).strip()

        return response.strip()

    def grade_answer(self, prediction: str, expected: str) -> bool:
        """
        Grade AIME answers by comparing the numerical values.
        AIME answers are integers from 0 to 999.
        """
        # Clean up prediction and expected to extract numbers
        pred_clean = re.sub(r'[^\d]', '', prediction)
        expected_clean = re.sub(r'[^\d]', '', expected)

        # If we couldn't extract a number, return False
        if not pred_clean or not expected_clean:
            return False

        # Convert to integers and compare
        try:
            pred_num = int(pred_clean)
            expected_num = int(expected_clean)
            return pred_num == expected_num
        except ValueError:
            return False

    def print_verbose_logs(
        self,
        idx: int,
        task_value: str,
        input: str,
        expected_output: str,
        prediction: str,
        score: int,
    ) -> str:
        steps = [
            f"Input:\n{input}",
            f"Score: {score}\nPrediction: {prediction}\nExpected Output: {expected_output}",
        ]
        verbose_logs = ""
        for i in range(len(steps) - 1):
            verbose_logs += steps[i]

            # don't add new line for penultimate step
            if i < len(steps) - 2:
                verbose_logs += " \n \n"

        if self.verbose_mode:
            print("*" * 50)
            print(f"Problem {idx + 1} (Task = {task_value})")
            print("*" * 50)
            print("")
            print(verbose_logs + f"\n \n{steps[-1]}")
            print("")
            print("=" * 70)

        return verbose_logs
