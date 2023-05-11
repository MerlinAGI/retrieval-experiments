import argparse

from huggingface_hub import snapshot_download
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM

def apply_delta(
        model_size: str = "7",
        target_model_path: str = None,
        base_model_path: str = None,
        delta_path: str = None,
):
    """Loads a llama model and applies the delta weights on top of it. Either specify the model size or all model paths.

    Args:
        model_size: The model size. Either 7 or 13
        target_model_path: default: weights/vicuna-{model_size}b
        base_model_path: default: decapoda-research/llama-{model_size}b-hf
        delta_path: default: lmsys/vicuna-{model_size}b-delta-v1.1
    """
    if target_model_path is None:
        target_model_path = f"weights/vicuna-{model_size}b"
    if base_model_path is None:
        base_model_path = f"decapoda-research/llama-{model_size}b-hf"
    if delta_path is None:
        delta_path = f"lmsys/vicuna-{model_size}b-delta-v1.1"

    print(f"Loading the base model from {base_model_path}")
    
    base = LlamaForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    base_tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_fast=False)

    print(f"Loading the delta from {delta_path}")
    delta = AutoModelForCausalLM.from_pretrained(
        delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=False
    )

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
  from jsonargparse import CLI

  CLI(apply_delta)
