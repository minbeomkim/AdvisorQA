from dataclasses import dataclass, field
from typing import Optional

import torch
import random
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, LlamaForCausalLM, TrainingArguments, AutoTokenizer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from trl import DPOTrainer

from typing import Dict, Optional

from accelerate import Accelerator 

accelerator = Accelerator()


# model_path = 'model/mistral/sft/final'
model_path = 'model/llama/sft/final'

peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )


# model_config = AutoConfig.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=peft_config, device_map={"": Accelerator().local_process_index}, load_in_4bit=True, trust_remote_code=True, is_trainable=True)
model = get_peft_model(model, peft_config)
model_ref = AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=peft_config, device_map={"": Accelerator().local_process_index}, load_in_4bit=True, trust_remote_code=True, is_trainable=False)
model_ref = get_peft_model(model_ref, peft_config)
model.print_trainable_parameters()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

import json, random

# Step 2: Load the dataset
before = []
with open('data/Advice_Collection/v3/train.json') as f:
# with open('data/ratio60/train.json') as f:
    for line in f:
        before.append(line)
before[0]

after = []
for _, item in enumerate(before):
    item = json.loads(item)
    if len(item['prefix']) == 4 :
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][2]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][-1]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][1], 'answer2': item['suffix'][2]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][1], 'answer2': item['suffix'][-1]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][2], 'answer2': item['suffix'][-1]})
        
    if len(item['prefix']) > 4 :
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][2]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][3]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][-1]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][1], 'answer2': item['suffix'][2]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][3]})        
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][1], 'answer2': item['suffix'][-1]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][2], 'answer2': item['suffix'][-1]})
        
random.shuffle(after)
train_dataset = after

before = []
with open('data/Advice_Collection/v3/test.json') as f:
    for line in f:
        before.append(line)
before[0]

after = []
for _, item in enumerate(before):
    item = json.loads(item)
    if len(item['prefix']) > 1:
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][1]})
        after.append({'prompt': 'Question: ' +  item['prefix'][0] + '\n\nAnswer: ', 'answer1': item['suffix'][0], 'answer2': item['suffix'][-1]})


test_dataset = after

def return_prompt_and_responses(samples) -> Dict[str, str]:
    return {
        "prompt": samples['prompt'],
        "chosen": samples["answer1"],   # rated better than k
        "rejected": samples["answer2"], # rated worse than j
    }

train_dataset = Dataset.from_list(train_dataset)
test_dataset = Dataset.from_list(test_dataset)

train_dataset = train_dataset.map(return_prompt_and_responses)
test_dataset = train_dataset.map(return_prompt_and_responses)

print(train_dataset)

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=384, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    remove_unused_columns=False,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    evaluation_strategy="steps",
    logging_first_step=True,
    logging_steps=10,  # match results in blog post
    eval_steps=5000,
    output_dir="model/llama/dpo-60:40",
    optim="paged_adamw_32bit",
    num_train_epochs=2,
    save_steps=1500,
#     max_steps=5000,
    bf16=True,
)

dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=128,
        max_target_length=384,
    )

dpo_trainer.train()
dpo_trainer.save_model('model/llama/dpo-60:40/final')