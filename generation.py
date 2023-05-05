import os
import torch
from datasets import load_from_disk
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import click
from pathlib import Path

def summarize(model, tokenizer, text, device, sample, max_length=512, num_beams=50):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True, max_length=max_length, truncation=True)
    if sample:
        with torch.no_grad():
            generated_ids = []
            for _ in range(num_beams):
                generated_id = model.generate(
                    input_ids=input_ids.to(device),
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1,
                    max_length=max_length
                ).flatten()
                generated_ids.append(generated_id)
    else:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids.to(device),
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=max_length
            )
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    torch.cuda.empty_cache()
    return preds

@click.command()
@click.option('--num_beams', default=50)
@click.option('--dataset_name', default="cnn-v2")
@click.option('--split', default="test")
@click.option('--model_name', default="t5-base-cnn")
@click.option('--save_iter', default=100)
@click.option('--sample', default=False)
@click.option('--start_iter', default=0)
def run(num_beams, dataset_name, split, model_name, save_iter, sample, start_iter):
    device = torch.device("cuda:0")
    
    if dataset_name == "xsum":
        dataset = load_from_disk("./pretrained/xsum")
        article_column = "document"
    elif dataset_name == "cnn-v2":
        dataset = load_from_disk("./pretrained/cnn-v2")
        article_column = "article"
    elif dataset_name == "paws":
        dataset = load_from_disk("./pretrained/paws")
        article_column = "sentence1"
    else:
        raise ValueError("Wrong dataset")
    
    if model_name == "t5-base-cnn":
        tokenizer = AutoTokenizer.from_pretrained("./pretrained/t5-base-cnn-tokenizer")
        model = AutoModelForSeq2SeqLM.from_pretrained("./pretrained/t5-base-cnn-model").to(device)
        text_transform = lambda text : text
    elif model_name == "t5-base-xsum":
        tokenizer = AutoTokenizer.from_pretrained("./pretrained/t5-base-xsum-tokenizer")
        model = AutoModelForSeq2SeqLM.from_pretrained("./pretrained/t5-base-xsum-model").to(device)
        text_transform = lambda text : "summarize: " + text
    elif model_name == "pegasus-xsum":
        tokenizer = AutoTokenizer.from_pretrained("./pretrained/pegasus-xsum-tokenizer")
        model = AutoModelForSeq2SeqLM.from_pretrained("./pretrained/pegasus-xsum-model").to(device)
        text_transform = lambda text : "summarize: " + text
    elif model_name == "pegasus-cnn":
        tokenizer = AutoTokenizer.from_pretrained("./pretrained/pegasus-cnn-tokenizer")
        model = AutoModelForSeq2SeqLM.from_pretrained("./pretrained/pegasus-cnn-model").to(device)
        text_transform = lambda text : text
    elif model_name == "t5-base-paws":
        tokenizer = AutoTokenizer.from_pretrained("./pretrained/t5-base-paws-tokenizer")  
        model = AutoModelForSeq2SeqLM.from_pretrained("./pretrained/t5-base-paws-model").to(device)
        text_transform = lambda text : "paraphrase: " + text + " </s>"
    elif model_name == "pegasus-paws":
        tokenizer = AutoTokenizer.from_pretrained("./pretrained/pegasus-paws-tokenizer")
        model = AutoModelForSeq2SeqLM.from_pretrained("./pretrained/pegasus-paws-model").to(device)
        text_transform = lambda text : text
    else:
        raise ValueError("Wrong model")
        
        
    with open(Path(__file__).parent / "generated" / f"{model_name}-tested-{dataset_name}-{split}-test-pkl-sample-{sample}.pkl", "wb") as file:
        pickle.dump([], file)
        
    
    summarized = []
    for it, item in enumerate(dataset[split][start_iter:]):
        i = it + start_iter
        summarized.append(summarize(model, tokenizer, text_transform(item[article_column]), device, sample, num_beams=num_beams))
        if (i + 1) % save_iter == 0:
            with open(Path(__file__).parent / "generated" / f"{model_name}-tested-{dataset_name}-{split}-sample-{sample}-iter-{i}.pkl", "wb") as file:
                pickle.dump(summarized, file)
        
    with open(Path(__file__).parent / "generated" / f"{model_name}-tested-{dataset_name}-{split}-sample-{sample}.pkl", "wb") as file:
        pickle.dump(summarized, file)
    
    
if __name__ == '__main__':
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
    run()