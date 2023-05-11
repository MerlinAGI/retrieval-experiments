import os
import sys

from typing import List
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

try:
    from .prompter import Prompter
except ImportError:
    from prompter import Prompter


device = "cuda"

def load_model_for_eval(
    load_8bit: bool = False,
    base_model: str = "weights/vicuna-7b",
    lora_weights: str = "adapter_weights/lora-400",
    use_finetuned: bool = True,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter()
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_finetuned:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=1,
        top_k=40,
        num_beams=6,
        max_new_tokens=150,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
    
    return evaluate
   

def load_model_from_size(model_size: str = "7",use_finetuned: bool = True):
    base_model = f"weights/vicuna-{model_size}b"
    if model_size == "7":
        lora_weights = f"adapter_weights/lora-400"
    else:
        lora_weights = f"adapter_weights/lora13-200"
    evaluate = load_model_for_eval(False, base_model, lora_weights, use_finetuned)
    return evaluate

def main(
    load_8bit: bool = False,
    base_model: str = "weights/vicuna-7b",
    lora_weights: str = "adapter_weights/lora-600",
    base_instruction: str = "Write a passage of a financial contract that answers the user's question",
):
    evaluate = load_model_for_eval(load_8bit, base_model, lora_weights)

    # input loop
    while True:
        user_input = input("\n\n\nInput: ")
        print("\n")
        print(evaluate(base_instruction, user_input))



if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
    