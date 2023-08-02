import pickle
import torch
import click
import numpy as np
from pathlib import Path
import pandas as pd
import multiprocessing

from nlgeval.metrics.mover_score import MoverScoreMetrics
from nlgeval.metrics.blanc import BlancMetrics
from nlgeval.metrics.ngram import BLEUMetrics, METEORMetrics, ROUGEMetrics, CHRFMetrics
from nlgeval.metrics.pretrained import BERTScoreMetrics, BARTScoreMetrics
from nlgeval.metrics.reference_free import CompressionMetrics, CoverageMetrics, LengthMetrics, NoveltyMetrics, DensityMetrics, RepetitionMetrics
from nlgeval.metrics.rouge_we import RougeWeMetrics
from nlgeval.metrics.s3 import S3Metrics
from nlgeval.metrics.statistical_measures import BaryScoreMetrics, DepthScoreMetrics, InfoLMMetrics

METRIC_NAME_TO_CLASS_AND_ARGS = dict(
       bary_score = (BaryScoreMetrics,
                     dict(
                          model_name = "./pretrained/blanc",
                          device = "cuda:0"
                     )),
       depth_score = (DepthScoreMetrics,
                     dict(
                          model_name = "./pretrained/blanc",
                          device = "cuda:0"
                     )),
       info_lm = (InfoLMMetrics,
                     dict(
                          model_name = "./pretrained/blanc",
                          device = "cuda:0"
                     )),
       mover_score = (MoverScoreMetrics,
                      dict(
                          n_gram=1,
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
       rouge_3 = (ROUGEMetrics, dict(setting="3")),
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
       rouge_we_1 = (RougeWeMetrics, dict(n_gram=1,
                                          emb_path="./pretrained/rouge_we"
                                          )),
       rouge_we_2 = (RougeWeMetrics, dict(n_gram=2,
                                          emb_path="./pretrained/rouge_we"
                                          )),
       rouge_we_3 = (RougeWeMetrics, dict(n_gram=3,
                                          emb_path="./pretrained/rouge_we"
                                          )),
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

ALL_METRICS = ["blanc_help", "blanc_tune", "compression", "coverage", "length", "novelty", "density", "repetition", "mover_score", "bleu", "rouge_1", "rouge_2", "rouge_l", "meteor", "chrf", "bertscore", "bartscore", "rouge_we_1", "rouge_we_2", "rouge_we_3", "s3_pyr", "s3_resp", "depth_score", "bary_score", "info_lm"]
CUDA_METRICS = ["blanc_help", "blanc_tune", "mover_score", "bertscore", "bartscore", "depth_score", "bary_score", "info_lm"]
CPU_METRICS = ["compression", "coverage", "length", "novelty", "density", "repetition", "bleu", "rouge_1", "rouge_2", "rouge_l", "meteor", "chrf", "rouge_we_1", "rouge_we_2", "rouge_we_3", "s3_pyr", "s3_resp"]
METRICS_WITH_INPUT = ["blanc_help", "blanc_tune", "compression", "coverage", "length", "novelty", "density", "repetition"]
METRICS_WITH_REF = ["mover_score", "bleu", "rouge_1", "rouge_2", "rouge_l", "meteor", "chrf", "bertscore", "bartscore", "rouge_we_1", "rouge_we_2", "rouge_we_3", "s3_pyr", "s3_resp", "depth_score", "bary_score", "info_lm"]
PATH_TO_DATA = dict(
    paraphrase = "data/paraphrase_data.csv",
    style_transfer = "data/style_transfer_data.csv",
    open_generation = "data/open_generation_data.csv",
    summarisation = "data/summarisation_data.csv",
)

@click.command()
@click.option('--data_name', default="summarisation")
@click.option('--batch_size', default=128)
@click.option('--cuda', default=False)
@click.option('--cpu', default=False)
@click.option('--start_with', default="blanc_help")
def run(data_name, batch_size, cuda, cpu, start_with):
    print("Reading dataset...")
    if data_name in PATH_TO_DATA:
        data = pd.read_csv(PATH_TO_DATA[data_name])
        generated = list(data.generated)
        if data_name == "summarisation":
            inputs = list(data.document)
            refs = list(data.summary)
        elif data_name == "open_generation":
            inputs = list(data.prompt)
        else:
            inputs = list(data.original)
    else:
        raise ValueError("Wrong dataset name")
           
    print("Measuring...")
    
    metrics_to_measure = ALL_METRICS
    if cuda:
        metrics_to_measure = CUDA_METRICS
    elif cpu:
        metrics_to_measure = CPU_METRICS
                
    started = False
    for metric_name in metrics_to_measure:
        if metric_name == start_with:
            started = True
        if not started:
            continue
        print(f"Instantiating {metric_name}...")
        metric_class, metric_args = METRIC_NAME_TO_CLASS_AND_ARGS[metric_name]
        metric = metric_class(**metric_args)
        if metric_name == "bertscore":
            metric.load()
        if metric_name == "depth_score":
            def foo():
                net = torch.nn.Sequential(
                    torch.nn.Linear(100, 300),
                    torch.nn.ReLU(),
                    torch.nn.Linear(300, 500),
                    torch.nn.ReLU(),
                    torch.nn.Linear(500, 100),
                    torch.nn.Flatten()
                ).cuda()
                i = 0
                A = torch.rand(1000, 100).cuda()
                criterion = torch.nn.MSELoss()
                while True:
                    if i % 1000 == 0:
                        A = torch.rand(1000, 100).cuda()
                    A = net(A)
                    loss = criterion(A, A)
                    loss.backward()
                    i += 1
            process = multiprocessing.Process(target=foo)
            process.start()
            
            
        print(f"Measuring with {metric_name}...")
            
        if metric_name in METRICS_WITH_INPUT:
            result = []
            for i in range(0, len(generated), batch_size):
                batch_len = min(batch_size, len(generated) - i)
                batch_gen = generated[i:i + batch_len]
                batch_inp = inputs[i:i + batch_len]
                cur_result = metric.evaluate_batch(batch_inp, batch_gen)
                result.append(cur_result)
            result = [value for sub_array in result for value in sub_array]
        elif metric_name in METRICS_WITH_REF:
            result = []
            for i in range(0, len(generated), batch_size):
                batch_len = min(batch_size, len(generated) - i)
                batch_gen = generated[i:i + batch_len]
                batch_ref = refs[i:i + batch_len]
                cur_result = metric.evaluate_batch(batch_gen, batch_ref)
                result.append(cur_result)
            result = [value for sub_array in result for value in sub_array]
        else:
            raise ValueError(f"We forgot about metric {metric_name} while creating METRICS_WITH_REF and METRICS_WITH_INPUT lists!")
            
        if metric_name == "depth_score":
            print(f"Terminating placeholder process...")
            process.terminate()
            
        print(f"Saving results of {metric_name}...")
        res_name = data_name + "-" + metric_name
        with open(f"results/{res_name}.pkl", "wb") as file:
            pickle.dump(result, file)
        torch.cuda.empty_cache()
    

if __name__ == '__main__':
    run()