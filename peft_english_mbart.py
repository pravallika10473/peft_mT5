import os
import sys
import time
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from peft import get_peft_config
from peft import get_peft_model
from peft import LoraConfig
from peft import TaskType
from peft import PeftModel
from peft import PeftConfig

from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import load_from_disk

import evaluate

os.environ["TRANSFORMERS_CACHE"]="/uusoc/exports/scratch/pra6/huggingface_cache"
os.environ["HF_DATASETS_CACHE"]="/uusoc/exports/scratch/pra6/huggingface_cache"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def configure_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = model.to(DEVICE)  # Move model to GPU if available
    return tokenizer, model

def preprocess_function(sample, args, padding="max_length"):
    model_checkpoint = args.model_name_or_path
    if model_checkpoint in ["google/mt5-small", "google/mt5-base", "google/mt5-large", "google/mt5-3b", "google/mt5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""
    inputs = [prefix+item for item in sample["text"]]
    # input_lengths = [len(tokenizer(item, truncation=True)["input_ids"]) for item in inputs]
    max_source_length = args.max_source_length
    # target length
    tokenized_targets = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["text", "summary"])
    target_lengths = [len(x) for x in tokenized_targets["input_ids"]]
    max_target_length = int(max(target_lengths))
    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    labels = tokenizer(sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def configure_lora_model(model, args):
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "shared", "lm_head"]
    config = LoraConfig(
        target_modules=target_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, config)
    return model

def print_trainable_parameters(model):
    all_params, trainable_params = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}")

def evaluate_peft_model(sample, tokenizer, model,max_target_length=50):
    # generate summary
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).to(DEVICE), do_sample=True, top_p=0.9, max_new_tokens=max_target_length)
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    # decode eval sample
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    return prediction, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="facebook/mbart-large-cc25")
    parser.add_argument("--dataset_path", type=str, default="csebuetnlp/xlsum")
    parser.add_argument("--language", type=str, default="english") 
    parser.add_argument("--max_source_length", type=int, default=1024) 

    # lora config
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # training config
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--per_device_batch_size", type=int, default=16)

    # logging and saving
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save_total_limit", type=int, default=5)

    parser.add_argument("--output_dir", type=str, default="output/mt5_peft_english")

    # misc
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--prediction_loss_only", action="store_true")

    args = parser.parse_args()
    model_checkpoint = args.model_name_or_path

    tokenizer, model = configure_model(args.model_name_or_path)

    raw_dataset = load_dataset(args.dataset_path, args.language)
    # tokenized_inputs = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]]).map(lambda x: tokenizer(x["text"], truncation=True), batched=True, remove_columns=["text", "summary"])
    # input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # max_source_length = int(max(input_lenghts))

    # The maximum total sequence length for target text after tokenization.
    # tokenized_targets = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["text", "summary"])
    # target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # max_target_length = int(max(target_lenghts))

    tokenized_dataset = raw_dataset.map(lambda x: preprocess_function(x, args), batched=True, remove_columns=["text", "summary", "id"])

    # save datasets to disk for later easy loading
    #tokenized_dataset["train"].save_to_disk("data/train")
    #tokenized_dataset["test"].save_to_disk("data/eval")

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    if args.use_lora:
        model = configure_lora_model(model=model, args=args)

    # saliency check on model parameters
    print_trainable_parameters(model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        auto_find_batch_size=False,
        per_device_train_batch_size=args.per_device_batch_size,
        learning_rate=args.lr, 
        num_train_epochs=args.epochs,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy=args.save_strategy,
        report_to="tensorboard",
        disable_tqdm=args.disable_tqdm,
        do_eval=args.do_eval,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        prediction_loss_only=args.prediction_loss_only,
        fp16=args.fp16  # Enable mixed precision training if available
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    model.config.use_cache = False  # silence the warnings.  re-enabling for inference!

    # train model
    trainer.train()
    output_dir = args.output_dir
    trainer.save_model(output_dir)
