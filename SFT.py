from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, LlamaForCausalLM, TrainingArguments, AutoTokenizer

from trl import SFTTrainer

# Step 1: Load the model
model_path = ''

peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )
token_num = '' # Insert your huggingface token

model_config = AutoConfig.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token=token_num, trust_remote_code=True)
# model_config = AutoConfig.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, config=model_config, load_in_4bit=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, token=token_num, device_map='auto', torch_dtype=torch.bfloat16, config=model_config, load_in_4bit=True, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, config=model_config, trust_remote_code=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

import json, random

# Step 2: Load the dataset
before = []
with open('data/Advice_Collection/v3/train.json') as f:
    for line in f:
        before.append(line)
        
after = []
for _, item in enumerate(before):
    item = json.loads(item)
    num = random.randrange(0,len(item['prefix']))
    text = str(item['prefix'][num])[:1000] + '\n\nAnswer: ' + str(item['suffix'][num])
    after.append({'text': text})
print(len(after))
after[0]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds_train = Dataset.from_list(after)
ds_train = ds_train.shuffle()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir="model/llama/sft",
    per_device_train_batch_size=8,
    logging_steps=100,
    num_train_epochs=5,
    save_steps=1000,
    learning_rate=5e-06,
    bf16=True
)

# Step 4: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    max_seq_length=512,
    tokenizer=tokenizer,
    peft_config=peft_config,
    dataset_text_field='text',
)

# Step 6: Save the model
trainer.train()
trainer.save_model('model/llama/sft/final')
