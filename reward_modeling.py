import pandas as pd
import numpy as np
import os, json, codecs, jsonlines, time, random
from transformers import pipeline, set_seed

import os

import random

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from trl import RewardTrainer
import json

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

before = []
with open('data/Advice_Collection/v3/train.json') as f:
    for line in f:
        before.append(line)

after = []
for _, item in enumerate(before):
    item = json.loads(item)

    # PL2
    if len(item['prefix']) == 2:
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]}) # hard
        # after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][-1]}) # easy

    # PL3
    if len(item['prefix']) == 3:
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][2]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][2]})

    # PL4
    if len(item['prefix']) == 4:
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][2]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][-1]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][2]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][-1]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][2], 'answer2': item['suffix'][-1]})

    # PL5
    if len(item['prefix']) > 4:
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][2]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][3]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][-1]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][2]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][3]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][-1]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][2], 'answer2': item['suffix'][3]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][2], 'answer2': item['suffix'][-1]})
        after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][3], 'answer2': item['suffix'][-1]})


    # # PL5
    # if len(item['prefix']) > 4:
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][2]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][3]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][4]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][-1]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][2]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][3]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][4]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][1], 'answer2': item['suffix'][-1]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][2], 'answer2': item['suffix'][3]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][2], 'answer2': item['suffix'][4]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][2], 'answer2': item['suffix'][-1]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][3], 'answer2': item['suffix'][4]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][3], 'answer2': item['suffix'][-1]})
    #     after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][4], 'answer2': item['suffix'][-1]})

print(len(after))

random.shuffle(after)
ds_train = Dataset.from_list(after)
ds_train.to_pandas()

before = []
with open('data/Advice_Collection/v3/test.json') as f:
    for line in f:
        before.append(line)

after = []
for _, item in enumerate(before):
    item = json.loads(item)

    after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]})
    after.append({'prompt': item['prefix'][0], 'answer1': item['suffix'][0], 'answer2': item['suffix'][-1]})
print(len(after))

ds_dev = Dataset.from_list(after)
ds_dev.to_pandas()

import utils.model_training.models.reward_model

# model_name = 'microsoft/deberta-v3-large'
# model_name = 'OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5'
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

model_name = 'EleutherAI/pythia-1.4b'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def formatting_func(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 1000, "return_tensors": "pt"}

    # for OpenAssistant
    prompt_plus_chosen_response = '<|prompter|>' + examples["prompt"] + '<|endoftext|><|assistant|>' + examples["answer1"] + '<|endoftext|>'
    prompt_plus_rejected_response = '<|prompter|>' + examples["prompt"] + '<|endoftext|><|assistant|>'  + examples["answer2"] + '<|endoftext|>'

    # # for pure RM
    # prompt_plus_chosen_response = 'Question: ' + examples["prompt"] + '\nAnswer: ' + examples["answer1"]
    # prompt_plus_rejected_response = 'Question: ' + examples["prompt"] + '\nAnswer: ' + examples["answer2"]

    # Then tokenize these modified fields.
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
    
    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }

ds_train = ds_train.map(formatting_func)
ds_dev = ds_dev.map(formatting_func)
ds_train = ds_train.shuffle()

from trl import RewardTrainer

training_args = TrainingArguments(
    output_dir="reward_model/PL5",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    evaluation_strategy="steps",
    logging_steps=250,
    learning_rate=5e-06,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=ds_train,
    eval_dataset=ds_dev,
)

trainer.train()

