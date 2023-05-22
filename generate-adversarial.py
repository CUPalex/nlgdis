import os
import pickle
import click
from pathlib import Path
from nltk import sent_tokenize, word_tokenize
import numpy as np
import string
from collections import defaultdict

def repete_words(summary):
    # 20% of all words in a new sentence are repetitions
    words = word_tokenize(summary)
    nopunct = [i for i, w in enumerate(words) if w not in string.punctuation]
    to_repeat = np.random.choice(nopunct, size=int(len(nopunct) * 0.25))
    for i, word in enumerate(words):
        words[i] = " ".join([words[i]] * (1 + np.count_nonzero(to_repeat == i)))
        if words[i] not in string.punctuation and i > 0:
            words[i] = " " + words[i]
    return "".join(words)

def change_word_order(summary):
    # 20% of all words in a new sentence changed their order
    words = word_tokenize(summary)
    nopunct = [i for i, w in enumerate(words) if w not in string.punctuation]
    num_to_change = int(len(nopunct) * 0.2)
    if num_to_change % 2 != 0:
        num_to_change += 1
    to_change = np.random.choice(nopunct, size=num_to_change, replace=False)
    for i in range(0, len(to_change), 2):
        words[to_change[i]], words[to_change[i + 1]] = words[to_change[i + 1]], words[to_change[i]]
    for i, word in enumerate(words):
        if words[i] not in string.punctuation and i > 0:
            words[i] = " " + words[i]
    return "".join(words)

def change_letters(summary):
    # 10% of letters changed
    nopunct = [i for i, s in enumerate(summary) if s in string.ascii_letters]
    to_misspell = np.random.choice(nopunct, size=int(len(nopunct) * 0.1), replace=False)
    s = [s for s in summary]
    for i in to_misspell:
        s[i] = string.ascii_lowercase[np.random.choice(len(string.ascii_lowercase))]
    return "".join(s)

def shuffle_sents(summary):
    sents = sent_tokenize(summary)
    order = np.random.permutation(len(sents))
    ordered_sents = [sents[i] for i in order]
    return " ".join(ordered_sents)

def add_random_info(summary, all_generated, k):
    i = np.random.choice([j for j in range(len(all_generated)) if j != k])
    random_summary = all_generated[i][np.random.choice(len(all_generated[i]))]
    random_summary_sents = sent_tokenize(random_summary)
    random_sent = random_summary_sents[np.random.choice(len(random_summary_sents))]
    return summary + " " + random_sent
def delete_sent(summary):
    sents = sent_tokenize(summary)
    to_delete = np.random.choice(len(sents))
    return " ".join(sents[:to_delete] + sents[to_delete + 1:])

def run():
    with open(Path(__file__).parent / "generated" / "t5-base-cnn-tested-cnn-v2-test-sample-True.pkl", "rb") as file:
        t5_cnn_sample = pickle.load(file)
        
    np.random.seed(42)
    grammar = []
    for summaries in t5_cnn_sample:
        grammar.append([])
        for summary in summaries:
            grammar[-1].append(repete_words(summary))
            grammar[-1].append(change_word_order(summary))
            grammar[-1].append(change_letters(summary))
    with open(Path(__file__).parent / "generated" / "grammar-adv-t5-cnn-sample.pkl", "wb") as file:
        pickle.dump(grammar, file)
        
        
    np.random.seed(42)
    coherence = []
    for summaries in t5_cnn_sample:
        coherence.append([])
        for summary in summaries:
            coherence[-1].append(shuffle_sents(summary))
    with open(Path(__file__).parent / "generated" / "coherence-adv-t5-cnn-sample.pkl", "wb") as file:
        pickle.dump(coherence, file)
        
    np.random.seed(42)
    relevance = []
    for i, summaries in enumerate(t5_cnn_sample):
        relevance.append([])
        for summary in summaries:
            relevance[-1].append(add_random_info(summary, t5_cnn_sample, i))
            relevance[-1].append(delete_sent(summary))
    with open(Path(__file__).parent / "generated" / "relevance-adv-t5-cnn-sample.pkl", "wb") as file:
        pickle.dump(relevance, file)
        
        
    
    
if __name__ == '__main__':
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
    run()