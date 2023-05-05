import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import click
from pathlib import Path

@click.command()
@click.option('--model_name', default=None)
@click.option('--scorer_name', default=None)
@click.option('--dataset_name', default=None)
@click.option('--datset_options', default=None)
@click.option('--save_dir', default="pretrained")
@click.option('--save_name', default="saved")
def run(dataset_name, scorer_name, datset_options, model_name, save_dir, save_name):
    if dataset_name is not None:
        dataset = load_dataset(dataset_name, datset_options)
        dataset.save_to_disk("./" + save_dir + "/" + save_name)
    elif model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained("./" + save_dir + "/" + save_name + "-model")
        tokenizer.save_pretrained("./" + save_dir + "/" + save_name + "-tokenizer")
    elif scorer_name is not None:
        if scorer_name == "grammar":
            tokenizer = RobertaTokenizer.from_pretrained("textattack/roberta-base-CoLA")
            model = RobertaForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA", num_labels=2)
            model.save_pretrained("./" + save_dir + "/grammar-scorer-model")
            tokenizer.save_pretrained("./" + save_dir + "/grammar-scorer-tokenizer")
        elif scorer_name == "relevance":
            tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
            model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
            model.save_pretrained("./" + save_dir + "/relevance-scorer-model")
            tokenizer.save_pretrained("./" + save_dir + "/relevance-scorer-tokenizer")
        elif scorer_name == "coherence":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            model = GPT2LMHeadModel.from_pretrained("gpt2-medium", return_dict=True)
            model.save_pretrained("./" + save_dir + "/coherence-scorer-model")
            tokenizer.save_pretrained("./" + save_dir + "/coherence-scorer-tokenizer")
        else:
            raise ValueError("Wrong scorer name")
    else:
        raise ValueError("Nothing to save")
    
    
if __name__ == '__main__':
    run()