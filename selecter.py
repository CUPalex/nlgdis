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
    def grammar_scorer(input_sent, generated_sent):
        # input_sent is not used
        input_ids = classifier_tokenizer_roberta_cola(generated_sent, max_length=512, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = classifier_roberta_cola(**input_ids).logits.flatten()
        score = torch.nn.functional.softmax(logits)[1].item()
        return score
    
    return grammar_scorer


def get_random_scorer():
    def random_scorer(input_sent, generated_sent):
        # input_sent, generated_sent are not used
        return np.random.rand()
    
    return random_scorer

def get_relevance_scorer(device):
    tokenizer = AutoTokenizer.from_pretrained("./pretrained/relevance-scorer-tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("./pretrained/relevance-scorer-model").to(device)

    results = []
    label_to_explanation = {0: "entailment", 1: "neutral", 2 : "contradiction"}
    
    # the more the better
    def relevance_scorer(input_sent, generated_sent):
        article_sents = sent_tokenize(input_sent)
        summary_sents = sent_tokenize(generated_sent)
        contradiction_scores = []
        for a_sent in article_sents:
            contradiction_scores.append([])
            for s_sent in summary_sents:
                tokenized_input_seq_pair = tokenizer.encode_plus(a_sent, s_sent,
                                                         max_length=256,
                                                         return_token_type_ids=True, truncation=True)
                input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
                token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
                attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_ids.to(device),
                                    attention_mask=attention_mask.to(device),
                                    token_type_ids=token_type_ids.to(device),
                                    labels=None)
                scores = torch.softmax(outputs[0], dim=1)[0].tolist()
                contradiction_scores[-1].append(scores[2])
        if len(contradiction_scores) > 0 and all([len(scores) > 0 for scores in contradiction_scores]):
            scores = [min([1 - score for score in scores_]) for scores_ in contradiction_scores]
            return sum(scores) / len(scores)
        return None
    return relevance_scorer

def get_coherence_scorer(device):
    tokenizer = GPT2Tokenizer.from_pretrained("./pretrained/coherence-scorer-tokenizer")
    model = GPT2LMHeadModel.from_pretrained("./pretrained/coherence-scorer-model", return_dict=True).to(device).eval()
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def perplexity(input_sent, generated_sent):
        # does not use input_sent
        max_length = model.config.n_positions
        stride=512
        inputs = tokenizer(generated_sent, return_tensors='pt')
        lls = []
        for i in range(0, inputs['input_ids'].size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, inputs['input_ids'].size(1))
            target_len = end_loc - i

            input_ids = inputs['input_ids'][:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone().to(device)

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs.loss * target_len
            lls.append(log_likelihood)
        return - torch.exp(torch.stack(lls).sum() / end_loc).item()
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
    scores = [[scorer(inp[input_column], r) for r in res] for res, inp in zip(results, dataset[split])]
    save_file = "{scorer}-{gen}".format(scorer=str(scorer_name), gen=str(generated_res_path).replace("/", "-"))
    with open(Path(__file__).parent / "scored" / "scores" / save_file, "wb") as file:
        pickle.dump(scores, file)
    
    print("Selecting best model...")
    selected = [res[np.argmax(score)] for score, res in zip(scores, results)]
    with open(Path(__file__).parent / "scored" / "selected" / save_file, "wb") as file:
        pickle.dump(selected, file)
    
if __name__ == '__main__':
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
    run()