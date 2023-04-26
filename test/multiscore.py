import evaluate
from tqdm import tqdm

s_bertscore = evaluate.load('bertscore')
s_sacrebleu = evaluate.load('sacrebleu')
s_ter = evaluate.load('ter')
s_parascore = evaluate.load('transZ/test_parascore')
s_meteor = evaluate.load('meteor')


import pandas as pd
df = pd.read_csv('data/result.csv')
# limit = 10
# cnt = 0
for _, row in tqdm(df.iterrows()):
    inp = row['Input']
    pred = row['Predict']
    # ref = row['Reference']
    # if cnt > limit:
        # break
    # cnt += 1
    # s_sacrebleu.add_batch(predictions=[pred], references=[ref])
    # s_ter.add_batch(predictions=[pred], references=[ref])
    # s_bertscore.add_batch(predictions=[pred], references=[inp])
    # s_parascore.add_batch(predictions=[pred], references=[inp])
    s_meteor.add_batch(predictions=[pred], references=[inp])

# res = s_sacrebleu.compute()
# print(f"SacreBLEU: {res['score']}")
# res = s_ter.compute()
# print(f"TER: {res['score']}")
# import numpy as np
# res = s_bertscore.compute(lang='en', model_type='microsoft/deberta-xlarge-mnli')
# print(f"BERTScore: {np.mean(res['f1'])}")
# res = s_parascore.compute()
# print(f"ParaScore: {np.mean(res['score'])}")
res = s_meteor.compute()
print(f"METEOR: {res['meteor']}")
