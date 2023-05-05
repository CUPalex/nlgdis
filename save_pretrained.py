import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import click
from pathlib import Path

@click.command()
@click.option('--model_name', default=None)
@click.option('--dataset_name', default=None)
@click.option('--datset_options', default=None)
@click.option('--save_dir', default="pretrained")
@click.option('--save_name', default="saved")
def run(dataset_name, datset_options, model_name, save_dir, save_name):
    if dataset_name is not None:
        dataset = load_dataset(dataset_name, datset_options)
        dataset.save_to_disk("./" + save_dir + "/" + save_name)
    elif model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained("./" + save_dir + "/" + save_name + "-model")
        tokenizer.save_pretrained("./" + save_dir + "/" + save_name + "-tokenizer")
    else:
        raise ValueError("Nothing to save")
    
    
if __name__ == '__main__':
    run()