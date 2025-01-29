from dataclasses import dataclass, field
from typing import Optional

import torch
import random
import tyro
from accelerate import Accelerator
from datasets import load_dataset, Dataset

from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, LlamaForCausalLM, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import utils.model_training.models.reward_model

# Step 1: Load the model

# model_path = 'model/sft/final'
model_path = 'model/llama/sft/final'

peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, 
                                                          device_map={"": Accelerator().local_process_index}, 
                                                        #   device_map='auto',
                                                          torch_dtype=torch.bfloat16, 
                                                          config=peft_config, 
                                                          load_in_4bit=True, 
                                                          trust_remote_code=True)
# model.print_trainable_parameters()
tokenizer.pad_token_id = tokenizer.eos_token_id

import json, random

# Step 2: Load the dataset
def create_and_prepare_dataset(data, tokenizer):
    dataset = Dataset.from_list(data)

    def tokenize(example):
        example["input_ids"] = tokenizer.encode(example["query"])[:128]
        example["query"] = tokenizer.decode(example["input_ids"])
        return example

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format("torch")
    return dataset

data = []
with open('data/Advice_Collection/v3/train.json') as f:
    for line in f:
        data.append({'query' : '[INST]: ' +  json.loads(line)['prefix'][0] + '[/INST]'})
        data.append({'query' : '[INST]: ' +  json.loads(line)['prefix'][0] + '[/INST]'})
        # data.append({"query" : json.loads(line)['prefix'][0]})
        # data.append({"query" : json.loads(line)['prefix'][0]})

random.shuffle(data)

data = create_and_prepare_dataset(data, tokenizer)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

config = PPOConfig(
    learning_rate=5e-6,
    batch_size=32,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    seed=0,
    use_score_scaling=False,
    use_score_norm=False,
    score_clip=None,
    kl_penalty = 'kl',
    init_kl_coef = 0.1,
    ppo_epochs=2,
)

# config = PPOConfig(
#     learning_rate=5e-6,
#     batch_size=1,
#     mini_batch_size=1,
#     gradient_accumulation_steps=1,
#     optimize_cuda_cache=True,
#     seed=0,
#     use_score_scaling=False,
#     use_score_norm=False,
#     score_clip=None,
#     ppo_epochs=1,
# )

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=data,
    data_collator=collator,
)

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 384,
}

model_name = 'reward_model/PL5/checkpoint-6000'
reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to('cuda')
reward_tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_score(model, tokenizer, prompt, response):
    inputs = tokenizer.encode_plus('<|prompter|>' + prompt + '<|endoftext|><|assistant|>' + response + '<|endoftext|>', truncation=True, padding="max_length", max_length=1024, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits
    logits = outputs.logits

    return logits.item()

harm_model_name = 'lifetox'
harm_model = AutoModelForSequenceClassification.from_pretrained(harm_model_name).to('cuda')
harm_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def harm_get_score(model, tokenizer, prompt, response):
    # Tokenize the input sequences
    inputs = tokenizer.encode_plus(prompt + " " + response, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to('cuda')
#     inputs = tokenizer.encode_plus(prompt + response, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(device)

    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits
    logits = outputs.logits

    return logits.item()

#### reward scaler ####
help_mean = -0.43565780
help_std = 1.40775449
harm_mean = 5.37234
harm_std = 7.30777

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    print(epoch)
    # print(len(batch["input_ids"]))
    # Get response from gpt2
    response_tensors= ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute score
    rewards = []
    for i in range(len(batch["query"])):
        helpf = torch.tensor((get_score(reward_model, reward_tokenizer, batch["query"][i],  batch["response"][i])-help_mean)/help_std).to('cuda')
        harmf = torch.tensor((harm_get_score(harm_model, harm_tokenizer, '',  batch["response"][i])-harm_mean)/harm_std).to('cuda')
        # print(helpf, harmf)
        rewards.append(helpf * 1.0 + harmf * 0.0)

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])

    if epoch % 150 == 0:
        ppo_trainer.save_pretrained(f'model/llama/ppo/{epoch}')

ppo_trainer.save_pretrained('model/llama/ppo/final')
