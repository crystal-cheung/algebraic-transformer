from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sglang as sgl
import pandas as pd
from datasets import load_dataset
import re
import ast

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
print(f"Test dataset (AIME 2023-2025): {len(test_data)} examples")

# Filter modular problems from training data
modular_problems = []
for idx, row in train_df.iterrows():
    if is_modular(row):
        modular_problems.append(row)

print(f"Found {len(modular_problems)} modular problems in training data")

# Process modular problems
transformed_results = []

for i, problem in enumerate(modular_problems[:5]):  # Process first 5 for testing
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
    
    if use_sglang:
        # For sglang, you might need to adjust this based on your sglang setup
        response = model.generate(prompt_content, sampling_params=sampling_params)['text']
    else:
        # Tokenize using chat format
        inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
        
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
        transformed_results.append({
            'original_problem': problem['Problem'],
            'symbolic_expressions': validated_result[0],
            'modulus': validated_result[1]
        })
    else:
        print("Failed to validate output format")

print(f"\nSuccessfully transformed {len(transformed_results)} problems")

# Display results
for i, result in enumerate(transformed_results):
    print(f"\nResult {i+1}:")
    print(f"Expressions: {result['symbolic_expressions']}")
    print(f"Modulus: {result['modulus']}")
