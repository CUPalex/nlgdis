import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BartForConditionalGeneration, BartTokenizer
import click
from pathlib import Path
from nlgeval.metrics.sources.rouge_we_utils import load_embeddings_from_web
from nlgeval.metrics.sources.s3_utils.s3_utils import load_model

@click.command()
@click.option('--model_name', default=None)
@click.option('--scorer_name', default=None)
@click.option('--dataset_name', default=None)
@click.option('--metric_name', default=None)
@click.option('--datset_options', default=None)
@click.option('--save_dir', default="pretrained")
@click.option('--save_name', default="saved")
def run(dataset_name, scorer_name, metric_name, datset_options, model_name, save_dir, save_name):
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
    elif metric_name is not None:
        if metric_name == "mover_score":
            tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI", do_lower_case=True)
            model = AutoModel.from_pretrained("textattack/bert-base-uncased-MNLI", output_hidden_states=True, output_attentions=True)
            model.save_pretrained("./" + save_dir + "/mover_score")
            tokenizer.save_pretrained("./" + save_dir + "/mover_score")
        elif metric_name == "blanc":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
            tokenizer.save_pretrained("./" + save_dir + "/blanc")
            model.save_pretrained("./" + save_dir + "/blanc")
        elif metric_name == "bart_score":
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
            tokenizer.save_pretrained("./" + save_dir + "/bart_score")
            model.save_pretrained("./" + save_dir + "/bart_score")
        elif metric_name == "rouge_we":
            load_embeddings_from_web(save_path="./" + save_dir + "/rouge_we")
        elif metric_name == "s3":
            load_model("./" + save_dir + "/s3_pyr", "https://drive.google.com/uc?id=19sdnH0e5YOtZBYi3kNQ-n0J8FRwuQhB2")
            load_model("./" + save_dir + "/s3_resp", "https://drive.google.com/uc?id=1qD-XfIiSocUi9QUR0ne4sn-vgmQSm3dV")
    else:
        raise ValueError("Nothing to save")
    
    
if __name__ == '__main__':
    run()