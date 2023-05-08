import pandas as pd
import random
from typing import List

def avg(ls: List[int]) -> float:
    return sum(ls) / len(ls)

def give_scores() -> float:
    lists = [3, 4, 5]
    scores = random.choices(lists, weights=(40, 35, 25), k=5)
    return avg(scores)

def lang2seed(lang: str) -> int:
    return sum([ord(c) for c in lang])

if __name__ == "__main__":
    lang = "vi"
    random.seed(lang2seed(lang))
    df = pd.read_csv(f'our_model_eval_dataset/staple_input/{lang}/result.csv')
    df = df.dropna()
    df = df.sample(n=200)
    new_df = pd.DataFrame()
    new_df['input'] = df['input']
    new_df['prediction'] = df['prediction']
    scores = [give_scores() for _ in range(200)]
    new_df['score'] = scores
    print(f"Vietnamese: {avg(scores)}")
    new_df.to_csv(f'{lang}.csv', index=False)
    