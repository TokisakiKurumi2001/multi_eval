import numpy as np

import evaluate
metrics = evaluate.load('ter')

import pandas as pd
# df = pd.read_csv('data/new.txt', names=["Output1", "Output2", "Output3", "Output4", "Truth"])
df = pd.read_csv('data/output_corrupted.csv')

final_scores = []
for _, row in df.iterrows():
    truth = row['Truth']
    arr = []
    for i in range(4):
        res = metrics.compute(predictions=[row[f"Output{i+1}"]], references=[[truth]])
        # print(f"{i+1}: {res['score']}")
        arr.append(res['score'])
    min_score = min(arr)
    # print(min_score)
    final_scores.append(min_score)

final_score = np.mean(final_scores)
print(final_score)