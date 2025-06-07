import re
import json
from typing import Optional, Any, List, Dict
import csv

log_file = "/projects/p32344/Reasoning/AIME2023/log.csv"



def extract_answer_from_response(response_text: str) -> Optional[Any]:
    """
    Extracts the answer from an LLM response, trying JSON parsing first,
    then keyword-based extraction.  Handles various answer types.

    Args:
        response_text: The raw text output from the LLM.

    Returns:
        The extracted answer (of any type) or None if extraction fails.
    """

    # Stage 1: Attempt JSON Parsing (Robust)
    json_match = re.search(r'\{\s*"answer"\s*:\s*(.*?)\s*\}', response_text, re.DOTALL | re.IGNORECASE)

    if json_match:
        try:
            json_string = json_match.group(0)
            json_data = json.loads(json_string)
            answer = json_data.get("answer")
            return answer  # Return the answer *without* type checking
        except json.JSONDecodeError:
            print(f"Invalid JSON found: {json_string}")
        except Exception as e:
            print(f"Other error in json parsing: {e}")
            
    # Stage 1.5: Attempt JSON Parsing (Nested Handling)
    json_match_nested = re.search(r'\{\s*"answer"\s*:\s*"(.*?)"\s*\}', response_text, re.DOTALL | re.IGNORECASE)
    if json_match_nested:
        try:
            json_string = json_match_nested.group(0)
            # Attempt to parse, but expect potential errors due to escaped chars
            json_data = json.loads(json_string)
            answer = json_data.get("answer")
            return answer
        except json.JSONDecodeError:
            print(f"Invalid JSON found (nested parsing): {json_string}")

    # Stage 2: Keyword-Based Extraction (Fallback, Generic)
    keyword_match = re.search(r'(?:^|\.|\?|\!)\s*answer(?:.*?)[:\s]\s*(.+?)(?:[\s.,;]|$)', response_text, re.IGNORECASE | re.MULTILINE)
    if keyword_match:
        answer = keyword_match.group(1).strip()
        return answer  

    # Stage 3: LaTeX Box Notation (More Precise)
    box_match = re.search(r'\\\[\s*\\boxed\{(?:\\text\{(.*?)\}|(.*?))\}\s*\\\]', response_text)
    if box_match:
        # Use the first non-None group (either with \text{} or without)
        answer = (box_match.group(1) or box_match.group(2)).strip()
        return answer

    # Stage 3.5: LaTeX Box Notation (Missing Closing)
    box_match_no_closing = re.search(r'\\boxed\{(?:\\text\{(.*?)\}|(.*?))\}', response_text)
    if box_match_no_closing:
        answer = (box_match_no_closing.group(1) or box_match_no_closing.group(2)).strip()
        return answer

    # Stage 4: Failure
    print(f"Failed to extract answer from response, see the log file")
    with open(log_file, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([response_text])
    return ""

def reward(queries: List[str], responses: List[str], labels: Optional[List[str]] = None) -> List[float]:
    """
    Reward function for the rule-based reward model.
    """
    rewards = []
    for query, response, label in zip(queries, responses, labels):
        answer = extract_answer_from_response(response)
        if answer == label:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

if __name__ == "__main__":
    file_path = "/projects/p32344/Reasoning/AIME2023/data/Qwen/Qwen3-1.7B/raw_input_output.csv"
    #read the Raw output from the file
    raw_output = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            raw_output.append(row[2])
            
    #extract the answer from the raw output
    for output in raw_output:
        # skip first row
        if output == "Raw Output":
            continue
        answer = extract_answer_from_response(output)
        # print(answer)
