import os
import json

import finetune.generate as generate


CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = f"{CACHE_DIR}/llama.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)
else:
    cache = {}


def complete(sys_promt, user_input, model="vicuna-7-finetuned", temp=0.0, max_tokens=None, n=1, return_list=False):
    
    model_parts = model.split("-")
    assert model_parts[0] == "vicuna"
    model_size = model_parts[1]
    is_finetuned = len(model_parts) > 2 and model_parts[2] == "finetuned"


    cache_key = json.dumps([model, sys_promt, user_input])
    if cache_key in cache:
        print("llama: using cached result")
        return cache[cache_key]
    print("llama: running completion")
    answers = generate.batch_generate([user_input], sys_promt, model_size, is_finetuned)
    
    if n == 1 and not return_list:
        res = answers[0]
    else:
        res = answers

    cache[cache_key] = res
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    return res
    