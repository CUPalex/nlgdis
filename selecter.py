import pickle
import torch
import click
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_grammar_scorer(device):
    classifier_tokenizer_roberta_cola = RobertaTokenizer.from_pretrained("./pretrained/grammar-scorer-tokenizer")
    classifier_roberta_cola = RobertaForSequenceClassification.from_pretrained("./pretrained/grammar-scorer-model", num_labels=2).to(device)
    
    # the more the better
    def grammar_scorer(input_sents, generated_sents):
        # input_sent is not used
        input_ids = classifier_tokenizer_roberta_cola(
            generated_sents, max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = classifier_roberta_cola(**input_ids).logits
        score = torch.nn.functional.softmax(logits, dim=-1).T[1].flatten().cpu().numpy()
        return list(score)

    
    return grammar_scorer


def get_random_scorer():
    def random_scorer(input_sents, generated_sents):
        # input_sent, generated_sent are not used
        return [np.random.rand() for _ in range(len(generated_sents))]
    
    return random_scorer

def get_relevance_scorer(device):
    tokenizer = AutoTokenizer.from_pretrained("./pretrained/relevance-scorer-tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("./pretrained/relevance-scorer-model").to(device)

    results = []
    label_to_explanation = {0: "entailment", 1: "neutral", 2 : "contradiction"}
    
    # the more the better
    def relevance_scorer(input_sents, generated_sents):
        contradiction_scores = []
        for input_sent, generated_sent in zip(input_sents, generated_sents):
            article_sents = sent_tokenize(input_sent)
            summary_sents = sent_tokenize(generated_sent)
            if len(article_sents) <= 0 or len(summary_sents) <= 0:
                return None

            all_possible_pairs = [
                (a_sent, s_sent) for a_sent in article_sents for s_sent in summary_sents
            ]
            tokenized_input_seq_pair = tokenizer.batch_encode_plus(all_possible_pairs,
                                                     max_length=256,
                                                     return_token_type_ids=True,
                                                     truncation=True,
                                                     padding=True)
            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long()
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long()
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long()

            with torch.no_grad():
                outputs = model(input_ids.to(device),
                                attention_mask=attention_mask.to(device),
                                token_type_ids=token_type_ids.to(device),
                                labels=None)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 2].flatten()

            contradiction_score = torch.min(scores.reshape(-1, len(summary_sents)), dim=-1).values.mean().item()
            contradiction_scores.append(contradiction_score)
        return contradiction_scores
    return relevance_scorer

def get_coherence_scorer(device):
    tokenizer = GPT2Tokenizer.from_pretrained("./pretrained/coherence-scorer-tokenizer")
    model = GPT2LMHeadModel.from_pretrained("./pretrained/coherence-scorer-model", return_dict=True).to(device).eval()
    tokenizer.pad_token = tokenizer.eos_token

    
    def count_loss(logits, inputs, attention_mask):
        labels = torch.nn.functional.one_hot(inputs.clone(), num_classes=50257)
        labels = (labels.permute(2, 0, 1) * attention_mask).permute(1, 2, 0)
        logits = (logits.permute(2, 0, 1) * attention_mask).permute(1, 2, 0)
        return  torch.log((torch.nn.functional.softmax(logits[:, :-1, :], dim=-1) * labels[:, 1:, :]).sum(dim=-1)
                         + (1 - attention_mask[:, 1:])).sum(dim=-1) / attention_mask.sum(dim=-1)

    def perplexity(input_sents, generated_sents):
        # does not use input_sent
        max_length = model.config.n_positions # 1024
        inputs = tokenizer(
            generated_sents, max_length=max_length, truncation=True, padding=True, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(inputs["input_ids"].to(device))
            return list(count_loss(
                outputs.logits.cpu(), inputs["input_ids"].cpu(), inputs["attention_mask"]).cpu().numpy())
    return perplexity


@click.command()
@click.option('--dataset_name', default="xsum")
@click.option('--split', default="test")
@click.option('--generated_res_path', default="generated/t5-base-xsum-tested-xsum-test.pkl")
@click.option('--scorer_name', default="random")
def run(dataset_name, split, generated_res_path, scorer_name):
    device = torch.device("cuda:0")
    
    print("Reading dataset...")
    if dataset_name == "xsum":
        dataset = load_from_disk("./pretrained/xsum")
        input_column = "document"
        ref_column = "summary"
    elif dataset_name == "cnn-v2":
        dataset = load_from_disk("./pretrained/cnn-v2")
        input_column = "article"
        ref_column = "highlights"
    elif dataset_name == "paws":
        dataset = load_from_disk("./pretrained/paws")
        input_column = "sentence1"
        ref_column = "sentence2"
    else:
        raise ValueError("Wrong dataset name")
        
    print("Reading generation results...")
    with open(Path(__file__).parent / generated_res_path, "rb") as file:
        results = pickle.load(file)
        
    print("Instantiating scorers...")
    if scorer_name == "grammar":
        scorer = get_grammar_scorer(device)
    elif scorer_name == "random":
        scorer = get_random_scorer()
    elif scorer_name == "relevance":
        scorer = get_relevance_scorer(device)
    elif scorer_name == "coherence":
        scorer = get_coherence_scorer(device)
    else:
        raise ValueError("Wrong scorer name")
        
    print("Scoring models...")
    save_file = "{scorer}-{gen}".format(scorer=str(scorer_name), gen=str(generated_res_path).replace("/", "-"))
    scores = []
    for i, (res, inp) in enumerate(zip(results, dataset[split])):
        scores.append(scorer([inp[input_column] for _ in range(len(res))], res))
        if (i + 1) % 200 == 0:
            with open(Path(__file__).parent / "scored" / "scores" / save_file + f"-iter{i}", "wb") as file:
                pickle.dump(scores, file)
    
    with open(Path(__file__).parent / "scored" / "scores" / save_file, "wb") as file:
        pickle.dump(scores, file)
    
    print("Selecting best model...")
    selected = [res[np.argmax(score)] for score, res in zip(scores, results)]
    with open(Path(__file__).parent / "scored" / "selected" / save_file, "wb") as file:
        pickle.dump(selected, file)
    
if __name__ == '__main__':
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
    run()