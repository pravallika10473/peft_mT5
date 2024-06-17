import os
import sys
import time
import random
import argparse

import numpy as np
from tqdm import tqdm
import torch

os.environ["TRANSFORMERS_CACHE"]="/uusoc/exports/scratch/pra6/huggingface_cache"
os.environ["HF_DATASETS_CACHE"]="/uusoc/exports/scratch/pra6/huggingface_cache"
os.environ["HF_HOME"] = "/uusoc/exports/scratch/pra6/huggingface_cache"


from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import DataCollatorForSeq2Seq

from peft import get_peft_config
from peft import get_peft_model
# from peft import LoraConfig
# from peft import TaskType
# from peft import PeftConfig

from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import load_from_disk

def load_peft_model(base_model, peft_model, dtype):
    from peft import PeftModel
    if dtype == "fp16":
        model = PeftModel.from_pretrained(base_model, peft_model, torch_dtype=torch.float16)
    
    else:
        raise Exception("unrecognized dtype")
    return model

def evaluate_peft_model(tokenizer, model, testset, generation_config, batch_size=16, n_samples=1000):
    num_batch = len(testset) // batch_size + 1 if len(testset) % batch_size != 0 else len(testset) // batch_size
    num_batch = min(num_batch, n_samples // batch_size+1)  # use a subset for faster evaluation

    predictions, references = [], []
    for batch_id in tqdm(range(num_batch)):
        batch = testset[batch_id*batch_size: (batch_id+1)*batch_size]
        tokenized_inputs = tokenizer(
            batch["text"], 
            return_tensors="pt",
            padding=True,
            truncation=True
            )
        outputs = model.generate(
            input_ids=tokenized_inputs["input_ids"].to(model.device),
            attention_mask=tokenized_inputs["attention_mask"].to(model.device),
            generation_config=generation_config,
        )
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        references.extend(batch["summary"])
    return predictions, references

def rouge_scorer(predictions, references, verbose=True):
    import evaluate
    metric = evaluate.load("rouge")
    res = metric.compute(
        predictions=predictions, 
        references=references, 
        use_stemmer=True,
        use_aggregator=True,
        )
    if verbose:
        print(f"rouge1: {res['rouge1']*100:.2f}%")
        print(f"rouge2: {res['rouge2']*100:.2f}%")
        print(f"rougeL: {res['rougeL']*100:.2f}%")
        print(f"rougeLsum: {res['rougeLsum']*100:.2f}%")
    return res

parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", type=str, default="facebook/mbart-large-cc25")
parser.add_argument("--language", type=str, default="english")
parser.add_argument("--use_peft", action="store_true")
parser.add_argument("--peft_model", type=str)
parser.add_argument("--dtype", type=str, default="fp16")

# dataset config
parser.add_argument("--n_samples", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=16)

# generation config
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--num_beams", type=int, default=1, help="use beam search by setting to >1")
parser.add_argument("--min_new_tokens", type=int, default=30)
parser.add_argument("--max_new_tokens", type=int, default=70)
parser.add_argument("--topk", type=int, default=50)
parser.add_argument("--topp", type=float, default=0.9)

args = parser.parse_args()

from peft_english_mbart import configure_model
tokenizer, model = configure_model(args.model_name_or_path)
if args.use_peft:
    assert args.peft_model != None, "need to specify a peft model"
    model = load_peft_model(base_model=model, peft_model=args.peft_model, dtype=args.dtype)

generation_config = GenerationConfig(
    min_new_tokens=args.min_new_tokens,
    max_new_tokens=args.max_new_tokens, 
    early_stopping=False,  # early stopping is only effective in beam search
    do_sample=args.do_sample,
    num_beams=args.num_beams,
    top_k=args.topk,
    top_p=args.topp,  
    bos_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)


dataset = load_dataset("csebuetnlp/xlsum", args.language)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)

predictions, references = evaluate_peft_model(
    tokenizer=tokenizer,
    model=model,
    testset=dataset["test"],
    generation_config=generation_config,
    batch_size=args.batch_size,
    n_samples=args.n_samples
)

rouge_scores = rouge_scorer(predictions, references)