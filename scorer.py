import pickle
import torch
import click
import numpy as np
from pathlib import Path
from datasets import load_from_disk

from nlgeval.metrics.mover_score import MoverScoreMetrics
from nlgeval.metrics.blanc import BlancMetrics
from nlgeval.metrics.ngram import BLEUMetrics, METEORMetrics, ROUGEMetrics, CHRFMetrics
from nlgeval.metrics.pretrained import BERTScoreMetrics, BARTScoreMetrics
from nlgeval.metrics.reference_free import CompressionMetrics, CoverageMetrics, LengthMetrics, NoveltyMetrics, DensityMetrics, RepetitionMetrics
from nlgeval.metrics.rouge_we import RougeWeMetrics
from nlgeval.metrics.s3 import S3Metrics


METRIC_NAME_TO_CLASS_AND_ARGS = dict(
       mover_score = (MoverScoreMetrics,
                      dict(
                          n_gram=2,
                          model_name="./pretrained/mover_score",
                          device="cuda:0",
                          batch_size=256
                     )),
       blanc_help = (BlancMetrics, dict(
                            model_name = "./pretrained/blanc",
                            device="cuda:0",
                            inference_batch_size = 256,
                            type= "help")
                    ),
       blanc_tune = (BlancMetrics, dict(
                            model_name = "./pretrained/blanc",
                            device="cuda:0",
                            inference_batch_size = 256,
                            finetune_batch_size = 256,
                            finetune_epochs = 5,
                            type= "tune")
                    ),
       bleu = (BLEUMetrics, {}),
       rouge_1 = (ROUGEMetrics, dict(setting="1")),
       rouge_2 = (ROUGEMetrics, dict(setting="2")),
       rouge_l = (ROUGEMetrics, dict(setting="L")),
       meteor = (METEORMetrics, {}),
       chrf = (CHRFMetrics, {}),
       bertscore = (BERTScoreMetrics, dict(device="cuda:0")),
       bartscore = (BARTScoreMetrics, dict(
                           checkpoint="./pretrained/bart_score",
                           device="cuda:0")
                   ),
       compression = (CompressionMetrics, {}),
       coverage = (CoverageMetrics, {}),
       length = (LengthMetrics, {}),
       novelty = (NoveltyMetrics, {}),
       density = (DensityMetrics, {}),
       repetition = (RepetitionMetrics, {}),
       rouge_we = (RougeWeMetrics, dict(emb_path="./pretrained/rouge_we")),
       s3_pyr = (S3Metrics, dict(
                           mode="pyr",
                           emb_path="./pretrained/rouge_we",
                           model_path="./pretrained/s3_pyr"
                   )),
       s3_resp = (S3Metrics, dict(
                           mode="resp",
                           emb_path="./pretrained/rouge_we",
                           model_path="./pretrained/s3_resp"
                   ))
)

METRICS_WITH_INPUT = ["blanc_help", "blanc_tune"]
METRICS_WITH_INPUT_REVERSED = ["compression", "coverage", "length", "novelty", "density", "repetition"]
METRICS_WITH_REF = ["mover_score", "bleu", "rouge_1", "rouge_2", "rouge_l", "meteor", "chrf", "bertscore", "bartscore", "rouge_we", "s3_pyr", "s3_resp"]

@click.command()
@click.option('--dataset_name', default="xsum")
@click.option('--split', default="test")
@click.option('--selected_res_path', default="scored/selected/random-generated-xsum-beam-search.pkl")
@click.option('--mover_score', default=False)
@click.option('--blanc_help', default=False)
@click.option('--blanc_tune', default=False)
@click.option('--bleu', default=False)
@click.option('--rouge_1', default=False)
@click.option('--rouge_2', default=False)
@click.option('--rouge_l', default=False)
@click.option('--meteor', default=False)
@click.option('--chrf', default=False)
@click.option('--bertscore', default=False)
@click.option('--bartscore', default=False)
@click.option('--compression', default=False)
@click.option('--coverage', default=False)
@click.option('--length', default=False)
@click.option('--novelty', default=False)
@click.option('--density', default=False)
@click.option('--repetition', default=False)
@click.option('--rouge_we', default=False)
@click.option('--s3_pyr', default=False)
@click.option('--s3_resp', default=False)
@click.option('--everything', default=False)
def run(dataset_name, split, selected_res_path, mover_score, blanc_help, blanc_tune, bleu, rouge_1, rouge_2, rouge_l, meteor, chrf,
       bertscore, bartscore, compression, coverage, length, novelty, density, repetition, rouge_we, s3_pyr, s3_resp, everything):
    
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
        
    print("Reading selected generation results...")
    with open(Path(__file__).parent / selected_res_path, "rb") as file:
        selected = pickle.load(file)
   
    print("Measuring...")
    selected_metrics = dict(
       mover_score=mover_score,
       blanc_help=blanc_help,
       blanc_tune=blanc_tune,
       bleu=bleu,
       rouge_1=rouge_1,
       rouge_2=rouge_2,
       rouge_L=rouge_l,
       meteor=meteor,
       chrf=chrf,
       bertscore=bertscore,
       bartscore=bartscore,
       compression=compression,
       coverage=coverage,
       length=length,
       novelty=novelty,
       density=density,
       repetition=repetition,
       rouge_we=rouge_we,
       s3_pyr=s3_pyr,
       s3_resp=s3_resp
    )
    if everything:
        for metric_name in selected_metrics.keys():
            selected_metrics[metric_name] = True
            
    for metric_name in selected_metrics.keys():
        if selected_metrics[metric_name]:
            print(f"Instantiating {metric_name}...")
            metric_class, metric_args = METRIC_NAME_TO_CLASS_AND_ARGS[metric_name]
            metric = metric_class(**metric_args)
            if metric_name == "bertscore":
                metric.load()
            
            print(f"Measuring with {metric_name}...")
            if metric_name in METRICS_WITH_REF:
                result = metric.evaluate_batch(selected, [dataset[split][i][ref_column] for i in range(len(dataset[split]))])
            elif metric_name in METRICS_WITH_INPUT:
                result = metric.evaluate_batch(selected, [dataset[split][i][input_column] for i in range(len(dataset[split]))])
            elif metric_name in METRICS_WITH_INPUT_REVERSED:
                result = metric.evaluate_batch([dataset[split][i][input_column] for i in range(len(dataset[split]))], selected)
            else:
                raise ValueError(f"We forgot about metric {metric_name} while creating METRICS_WITH_REF and METRICS_WITH_INPUT lists!")
                
            print(f"Saving results of {metric_name}...")
            res_name = selected_res_path.replace("/", "-")
            with open(f"./metrics-results/{metric_name}_{res_name}.pkl", "wb") as file:
                pickle.dump(result, file)
            torch.cuda.empty_cache()
    

    
    
if __name__ == '__main__':
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
    run()