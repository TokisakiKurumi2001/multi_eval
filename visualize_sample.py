import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_chart(data, name, filename):
    langs = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(langs, values, color ='blue', width = 0.4)
    plt.xlabel("Languages")
    plt.ylabel("Number of samples")
    plt.title(name)
    plt.savefig(filename)

if __name__ == "__main__":
    path = "baseline_eval_dataset/wmt19_input"
    langs = os.listdir(path)
    stat = {}
    for lang in langs:
        df = pd.read_csv(f'{path}/{lang}/result.csv')
        stat[lang] = len(df)
    
    stat = dict(sorted(stat.items(), key=lambda x:x[1]))
    plot_chart(stat, "WMT19 data statistics", 'wmt19.png')