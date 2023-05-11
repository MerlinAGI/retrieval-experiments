import openai
import dotenv
import os
import json

import llm_finetuned

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = f"{CACHE_DIR}/llm_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)
else:
    cache = {}


def complete(sys_promt, user_input, model="gpt-4", temp=0.0, max_tokens=None, n=1, return_list=False, ignore_cache=False):
    if model=="turbo":
        model = "gpt-3.5-turbo"
    elif model=="gpt-4":
        pass
    else:
        return llm_finetuned.complete(sys_promt, user_input, model, max_tokens=max_tokens, return_list=return_list, ignore_cache=ignore_cache)
    
    messages = [
        {"role": "system", "content": sys_promt},
        {"role": "user", "content": user_input},
    ]
    cache_key = json.dumps([model, messages, n])
    if cache_key in cache:
        print("using cached result")
        return cache[cache_key]
    print("running completion")
    
    r = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        n=n,
    )
    if n == 1 and not return_list:
        res = r.choices[0]["message"]["content"]
    else:
        res = [choice["message"]["content"] for choice in r.choices]

    cache[cache_key] = res
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    return res

def complete_json(sys_promt, user_input, model="gpt-4"):
    res = complete(sys_promt, user_input, model=model)
    try:
        return json.loads(res)
    except json.decoder.JSONDecodeError:
        print("Error decoding JSON:")
        print(res)
        return None
    