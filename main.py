from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sglang as sgl
import pandas as pd
from datasets import load_dataset
import re
import ast
from algebraic_transformer import load_algebraic_transformer_if_exists
import numpy as np

def is_modular(example):
    """Filter modular arithmetic problems"""
    text = example['Problem'].lower()
    return any(term in text for term in ['mod', 'remainder', 'congru', 'divided by'])

def create_prompt_template(problem_text):
    """Create prompt template for symbolic expression extraction"""
    return f"""
        Extract symbolic expressions from the natural language math statements below.
        Output a Python list of tuples [(variable, expression), ...] and the modulus as an integer.

        Example 1:
        Input: Let x be 2 mod 5. Let y = x + 3. Is y equal to 0 mod 5?
        Output: [("x", 2), ("y", "x + 3"), ("y", 0)], 5

        Example 2:
        Input: Let x be 4 mod 7. Let y = 2 * x. Is y equal to 1 mod 7?
        Output: [("x", 4), ("y", "2 * x"), ("y", 1)], 7

        Now extract from this input:
        Input: {problem_text}
        Output:
        """

def validate_and_fix_output(output_text):
    """Validate and fix the transformation result to match expected format"""
    try:
        # Clean the output text
        output_text = output_text.strip()
        
        # Try to extract the part after "Output:" if it exists
        if "Output:" in output_text:
            output_text = output_text.split("Output:")[-1].strip()
        
        # Remove any markdown formatting
        output_text = re.sub(r'```.*?```', '', output_text, flags=re.DOTALL)
        output_text = output_text.replace('`', '')
        
        # Try to parse as Python literal
        try:
            result = ast.literal_eval(output_text)
            if isinstance(result, tuple) and len(result) == 2:
                expressions, modulus = result
                if isinstance(expressions, list) and isinstance(modulus, int):
                    return result
        except:
            pass
        
        # If direct parsing fails, try to extract using regex
        pattern = r'\[(.*?)\],\s*(\d+)'
        match = re.search(pattern, output_text)
        if match:
            expressions_str = match.group(1)
            modulus = int(match.group(2))
            
            # Parse expressions
            expressions = []
            expr_pattern = r'\("([^"]+)",\s*([^)]+)\)'
            for expr_match in re.finditer(expr_pattern, expressions_str):
                var = expr_match.group(1)
                expr = expr_match.group(2).strip()
                # Remove quotes if present
                if expr.startswith('"') and expr.endswith('"'):
                    expr = expr[1:-1]
                elif expr.startswith("'") and expr.endswith("'"):
                    expr = expr[1:-1]
                else:
                    # Try to convert to int if possible
                    try:
                        expr = int(expr)
                    except:
                        pass
                expressions.append((var, expr))
            
            return (expressions, modulus)
        
        return None
    except Exception as e:
        print(f"Error validating output: {e}")
        return None

def parse_ground_truth_answer(answer):
    """
    Parse the ground truth answer from the dataset.
    
    Args:
        answer: Answer from the dataset (could be string or number)
    
    Returns:
        parsed_answer: Integer answer or None if parsing fails
    """
    try:
        # If it's already a number
        if isinstance(answer, (int, float)):
            return int(answer)
        
        # If it's a string, try to extract the number
        if isinstance(answer, str):
            # Remove common prefixes/suffixes
            answer = answer.strip()
            
            # Extract numbers from the string
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return int(numbers[0])  # Take the first number found
            
            # Handle special cases like "Yes"/"No" for boolean questions
            if answer.lower() in ['yes', 'true']:
                return 1
            elif answer.lower() in ['no', 'false']:
                return 0
        
        return None
    except Exception as e:
        print(f"Error parsing answer '{answer}': {e}")
        return None

def compute_with_algebraic_transformer(model, expressions, modular, seq_len=10):
    """
    Use algebraic transformer to compute results.
    
    Args:
        model: Trained algebraic transformer
        expressions: List of symbolic expressions
        modular: The modular base
        seq_len: Length of input sequence
    
    Returns:
        transformer_result: Model prediction
        confidence: Confidence score
    """
    try:
        # Convert expressions to numerical sequence
        sequence = []
        
        # Extract numerical values from expressions
        for var, expr in expressions:
            if isinstance(expr, (int, float)):
                sequence.append(int(expr) % modular)
            else:
                # For string expressions, try to extract numbers
                numbers = re.findall(r'\d+', str(expr))
                if numbers:
                    sequence.extend([int(n) % modular for n in numbers])
        
        # Pad or truncate to desired length
        if len(sequence) < seq_len:
            sequence.extend([0] * (seq_len - len(sequence)))
        else:
            sequence = sequence[:seq_len]
        
        # Create batch (single example)
        input_sequences = [sequence]
        
        # Get model prediction
        with torch.no_grad():
            logits = model(input_sequences)
            probabilities = torch.sigmoid(logits)
            prediction = (logits > 0).long().item()
            confidence = probabilities.item() if prediction == 1 else (1 - probabilities.item())
        
        return prediction, confidence
    
    except Exception as e:
        print(f"Error computing with algebraic transformer: {e}")
        return None, 0.0

def check_correctness(result, ground_truth, modular=None):
    """
    Check if result matches ground truth.
    
    Args:
        result: Computed result
        ground_truth: Ground truth answer
        modular: Modular base (for modular comparison if needed)
    
    Returns:
        is_correct: Boolean indicating correctness
    """
    if result is None or ground_truth is None:
        return False
    
    if modular is not None:
        # For modular arithmetic, compare modulo the base
        return (result % modular) == (ground_truth % modular)
    else:
        return result == ground_truth

def process_modular_problem_with_transformer(problem_data, ground_truth_answer, model_dir="./models"):
    """
    Process a single modular problem with algebraic transformer only.
    
    Args:
        problem_data: Dictionary with symbolic expressions and modulus
        ground_truth_answer: Ground truth answer from dataset
        model_dir: Directory containing saved models
    
    Returns:
        result_dict: Complete analysis results
    """
    expressions = problem_data['symbolic_expressions']
    modular = problem_data['modulus']
    
    # Parse ground truth answer
    parsed_ground_truth = parse_ground_truth_answer(ground_truth_answer)
    
    # Load corresponding algebraic transformer
    model, config = load_algebraic_transformer_if_exists(modular, model_dir)
    
    # Use algebraic transformer to compute result
    transformer_result = None
    confidence = 0.0
    model_available = model is not None
    
    if model is not None:
        transformer_result, confidence = compute_with_algebraic_transformer(
            model, expressions, modular
        )
    
    # Check correctness against ground truth
    transformer_correct = check_correctness(transformer_result, parsed_ground_truth, modular) if transformer_result is not None else False
    
    result_dict = {
        'modular': modular,
        'expressions': expressions,
        'ground_truth': {
            'raw_answer': ground_truth_answer,
            'parsed_answer': parsed_ground_truth
        },
        'transformer_evaluation': {
            'result': transformer_result,
            'confidence': confidence,
            'model_available': model_available,
            'correct': transformer_correct
        },
        'final_answer': transformer_result,
        'is_correct': transformer_correct
    }
    
    return result_dict

# Configuration
use_sglang = False  # Set this based on your preference
dp_size = 1
tp_size = 1

# Use Qwen chat model
model_name = "Qwen/Qwen3-1.7B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if use_sglang:
    model = sgl.Engine(
        model_path=model_name,
        dp_size=dp_size,
        tp_size=tp_size,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

sampling_params = {
    "max_new_tokens": 200,
    "temperature": 0.0
}

# Load dataset
dataset = load_dataset("czl9794/AIME_1983-2025")
train_df = pd.DataFrame(dataset['train'])
print(f"Train dataset (1983-2022): {len(train_df)} examples")

test_data = dataset['test']
test_df = pd.DataFrame(test_data)
print(f"Test dataset (AIME 2023-2025): {len(test_data)} examples")

# Filter modular problems from testing data
modular_problems = []
for idx, row in test_df.iterrows():
    if is_modular(row):
        modular_problems.append(row)

print(f"Found {len(modular_problems)} modular problems in testing data")

# Process modular problems
transformed_results = []

for i, problem in enumerate(modular_problems):  # Process first 5 for testing
    print(f"\nProcessing problem {i+1}:")
    print(f"Original: {problem['Problem'][:200]}...")
    
    # Create prompt
    prompt_content = create_prompt_template(problem['Problem'])
    
    chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant that extracts symbolic math expressions."
        },
        {
            "role": "user",
            "content": prompt_content
        }
    ]

    # Tokenize using chat format
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)

    if use_sglang:
        # For sglang, you might need to adjust this based on your sglang setup
        response = model.generate(inputs, sampling_params=sampling_params)['text']
    else:
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=200,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
    
    print(f"Raw model output: {response}")
    
    # Validate and fix the output
    validated_result = validate_and_fix_output(response)
    
    if validated_result:
        print(f"Validated result: {validated_result}")
        problem_data = {
            'original_problem': problem['Problem'],
            'symbolic_expressions': validated_result[0],
            'modulus': validated_result[1]
        }

        # Process with algebraic transformer and ground truth
        complete_result = process_modular_problem_with_transformer(
            problem_data, problem['Answer']
        )
        transformed_results.append(complete_result)

        # Display detailed results
        print(f"\n=== Analysis for Problem {i+1} ===")
        print(f"Ground Truth: {complete_result['ground_truth']['parsed_answer']}")
        print(f"Modular: Z_{complete_result['modular']}")
        
        if complete_result['transformer_evaluation']['model_available']:
            print(f"Transformer Result: {complete_result['transformer_evaluation']['result']} ({'✓' if complete_result['transformer_evaluation']['correct'] else '✗'})")
            print(f"Confidence: {complete_result['transformer_evaluation']['confidence']:.3f}")
        else:
            print(f"Transformer Result: Model not available")
        
        print(f"Final Answer: {complete_result['final_answer']}")
        print(f"Overall Correct: {'✓' if complete_result['is_correct'] else '✗'}")

    else:
        print("Failed to validate output format")

print(f"\nSuccessfully transformed {len(transformed_results)} problems")

# Summary statistics
if transformed_results:
    total_problems = len(transformed_results)
    problems_with_transformer = sum(1 for r in transformed_results if r['transformer_evaluation']['model_available'])
    transformer_correct = sum(1 for r in transformed_results if r['transformer_evaluation']['correct'])
    
    print(f"\n=== Summary Statistics ===")
    print(f"Total problems processed: {total_problems}")
    print(f"Problems with available transformer: {problems_with_transformer}")
    
    if problems_with_transformer > 0:
        print(f"Transformer accuracy: {transformer_correct}/{problems_with_transformer} ({transformer_correct/problems_with_transformer*100:.1f}%)")
        
        # Average confidence for correct vs incorrect predictions
        correct_confidences = [r['transformer_evaluation']['confidence'] for r in transformed_results 
                             if r['transformer_evaluation']['model_available'] and r['transformer_evaluation']['correct']]
        incorrect_confidences = [r['transformer_evaluation']['confidence'] for r in transformed_results 
                               if r['transformer_evaluation']['model_available'] and not r['transformer_evaluation']['correct']]
        
        if correct_confidences:
            print(f"Average confidence (correct): {np.mean(correct_confidences):.3f}")
        if incorrect_confidences:
            print(f"Average confidence (incorrect): {np.mean(incorrect_confidences):.3f}")
    else:
        print("No transformer models available for any problems")
    
    # Detailed breakdown
    print(f"\n=== Results Breakdown ===")
    for i, result in enumerate(transformed_results):
        if result['transformer_evaluation']['model_available']:
            status = "✓" if result['is_correct'] else "✗"
            conf = result['transformer_evaluation']['confidence']
            print(f"Problem {i+1} {status}: GT={result['ground_truth']['parsed_answer']}, "
                  f"Transformer={result['transformer_evaluation']['result']} (conf: {conf:.3f})")
        else:
            print(f"Problem {i+1} ?: GT={result['ground_truth']['parsed_answer']}, "
                  f"Transformer=N/A (no model)")
