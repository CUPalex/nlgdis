import os
import torch
from datasets import load_from_disk
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import click
from pathlib import Path
from torch.utils.data import DataLoader

def summarize(model, tokenizer, dataset, article_column, device, sample, max_length=512, num_beams=30):
    dataset = dataset.map(lambda item: {article_column : tokenizer(
        list(item[article_column]), return_tensors="pt", add_special_tokens=True, max_length=max_length, truncation=True
    )["input_ids"]}, batched=True, batch_size=16)
    
    dataloader = DataLoader(dataset, batch_size=4)
    
    if sample:
        with torch.no_grad():
            res = []
            for batch in dataloader:
                generated_ids = model.generate(
                    input_ids=torch.vstack(batch[article_column]).T.to(device),
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=num_beams,
                    max_length=max_length
                )
                res.extend([list(gen) for gen in generated_ids])
    else:
        with torch.no_grad():
            res = []
            for batch in dataloader:
                generated_ids = model.generate(
                    input_ids=torch.vstack(batch[article_column]).T.to(device),
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    max_length=max_length
                )
                res.extend([list(gen) for gen in generated_ids])
            
    preds = tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
        
    def transform(item):
        item[article_column] = text_transform(item[article_column])
        return item
        
        
    with open(Path(__file__).parent / "generated" / f"{model_name}-tested-{dataset_name}-{split}-test-pkl-sample-{sample}.pkl", "wb") as file:
        pickle.dump([], file)
        
    
    dataset = dataset[split].select(list(range(start_iter, len(dataset[split])))).map(transform)
    summarized = summarize(model, tokenizer, dataset, article_column, device, sample, num_beams=num_beams)
        
    with open(Path(__file__).parent / "generated" / f"{model_name}-tested-{dataset_name}-{split}-sample-{sample}.pkl", "wb") as file:
        pickle.dump(summarized, file)
    
    
if __name__ == '__main__':
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
    run()