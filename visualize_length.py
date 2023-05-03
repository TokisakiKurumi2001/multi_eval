import pandas as pd
import numpy as np
import os
import jieba
from konoha import WordTokenizer
import matplotlib.pyplot as plt

from typing import List

tokenizer = WordTokenizer('MeCab')

def plot_hist(data, name, filename):
    fig = plt.figure(figsize=(10, 5))
    plt.hist(data, bins=10)
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.title(name)
    plt.savefig(filename)

def get_len(sent, lang) -> int:
    if lang == "zh":
        return len([i for i in jieba.cut(sent, cut_all=True)])
    elif lang == "ja":
        return len(tokenizer.tokenize(sent))
    else:
        return len(sent.split(" "))

def extract_len(ls, lang) -> List[int]:
    return list(map(lambda x: get_len(x, lang), ls))

if __name__ == "__main__":
    path = "baseline_eval_dataset/staple_input"
    langs = os.listdir(path)
    lengths = []
    for lang in langs:
        df = pd.read_csv(f'{path}/{lang}/result.csv')
        lengths.extend(extract_len(df['input'].values.tolist(), lang))
    
    plot_hist(lengths, "Staple number of words per sample histogram", "staple_hist.png")