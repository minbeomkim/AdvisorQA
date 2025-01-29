import pandas as pd
import numpy as np
import os, json, codecs, jsonlines, time
from transformers import pipeline, set_seed

import os

import random

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoConfig,
)
from trl import RewardTrainer
import json
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, PeftConfig

import utils.model_training.models.reward_model

token_num = '' # Insert your huggingface token

path = "model/llama/dpo/final"

LLM_config = PeftConfig.from_pretrained(path, local_files_only=True)

LLM = AutoPeftModelForCausalLM.from_pretrained(
    path,
    device_map='auto',
    token='token_num',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    config=LLM_config,
    local_files_only=True,
    load_in_4bit=True,
)
LLM_tokenizer = AutoTokenizer.from_pretrained(path, token='token_num', trust_remote_code=True)


test = []
with open('data/Advice_Collection/v3/test.json') as f:
    for line in f:
        test.append(line)
json.loads(test[0])

def get_score(model, tokenizer, prompt, response):
    inputs = tokenizer.encode_plus('<|prompter|>' + prompt + '<|endoftext|><|assistant|>' + response + '<|endoftext|>', truncation=True, padding="max_length", max_length=1024, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits
    logits = outputs.logits

    return logits.item()

LLM_tokenizer.pad_token = LLM_tokenizer.eos_token

generation = []
score = []
for i in range(1000):
    print(i)
    prefix = "Question: " + json.loads(test[i])['prefix'][0] + "\n\nAnswer: "
    inputs = LLM_tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to('cuda')
    # Perform forward pass
    with torch.no_grad():
        outputs = LLM.generate(**inputs, do_sample=True, top_p=0.92, max_new_tokens=512, pad_token_id=LLM_tokenizer.eos_token_id)
        out=LLM_tokenizer.batch_decode(outputs[:, len(inputs.input_ids[0]):], skip_special_tokens=True)
    # print(out[0]) 

    generation.append(out[0])
    # score.append(get_score(model, tokenizer, prefix, out))

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

with open('results/results.jsonl', 'w', encoding = 'utf-8') as out:
    for i in range(len(generation)):
        jout = json.dumps({'prefix': json.loads(test[i])['prefix'][0], 'suffix': generation[i]}, cls=MyEncoder) + '\n'
        out.write(jout)

