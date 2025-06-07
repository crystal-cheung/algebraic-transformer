from AIME import AIMEEval
import argparse
import sglang as sgl
from deepeval.models import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import pandas as pd
import os
from sglang.srt.server_args import PortArgs, ServerArgs

class EvalModel(DeepEvalBaseLLM):
    def __init__(self, 
                 model_path: str,
                 use_sglang: bool = False,
                 max_new_tokens: int = 4096,
                 temperature: float = 0.0,
                 dp_size: int = 4,
                 tp_size: int = 1,
                 tokenizer: str = None):
        if 'saves' not in model_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer
            )
        if use_sglang:
            self.model = sgl.Engine(
                model_path=model_path,
                dp_size=dp_size,
                tp_size=tp_size,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto"
            )

        self.tokenizer = tokenizer
        self.use_sglang = use_sglang
        self.model_path = model_path
        self.sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
        

    def load_model(self):
        return self.model
    
    def template_prompt(self, prompt: str) -> str:
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return text

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        template_prompt = self.template_prompt(prompt)
        if self.use_sglang:
            response = model.generate(template_prompt, sampling_params=self.sampling_params)['text']

        return response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def batch_generate(self, promtps: List[str]) -> List[str]:
        model = self.load_model()
        template_prompts = [self.template_prompt(prompt) for prompt in promtps]
        if self.use_sglang:
            responses = model.generate(template_prompts, sampling_params=self.sampling_params)
            responses = [response['text'] for response in responses]

        return responses

    def get_model_name(self):
        return self.model_path
    
    
    
def main(args):
    try:
        benchmark = AIMEEval(tasks=args.task)

        # load the model
        eval_model = EvalModel(args.model, use_sglang=True, dp_size=args.dp_size, tp_size=args.tp_size, tokenizer=args.tokenizer)
        benchmark.evaluate(model=eval_model, batch_size=32)
        print(benchmark.overall_score)

        # Create output directory if it doesn't exist
        if 'saves' not in args.model:
            output_dir = f"./data/{args.model}"
        else:
            # Extract the path after 'saves/' and place it under ./data/
            output_dir = f"./data/{'/'.join(args.model.split('saves/')[1:])}"
        os.makedirs(output_dir, exist_ok=True)
        
        # export the predictions and task scores
        predictions = benchmark.predictions
        task_scores = benchmark.task_scores
        raw_input_output = benchmark.raw_input_output   
        predictions.to_csv(f"{output_dir}/predictions.csv", index=False)
        task_scores.to_csv(f"{output_dir}/task_scores.csv", index=False)
        raw_input_output.to_csv(f"{output_dir}/raw_input_output.csv", index=False)
        print(f"Results saved to {output_dir}/predictions.csv, {output_dir}/task_scores.csv and {output_dir}/raw_input_output.csv")

    finally:
        # Cleanup SGLang engine
        if 'eval_model' in locals() and hasattr(eval_model, 'model'):
            del eval_model.model
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model", type=str, default="saves/dpo_aime/merged")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-1.7B", help="Only used for llama-factory models since they have a different tokenizer after training")
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--task", type=str, default=['train', 'test'])
    args = parser.parse_args()
    main(args)
