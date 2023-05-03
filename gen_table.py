import pandas as pd
import re
from typing import List
import numpy as np

class Parser:
    def __call__(self, doc: List[str]):
        arr = []
        first_time = True
        dictionary = None
        for line in doc:
            attr, value = line.split(": ")
            if attr == "Lang":
                if first_time:
                    first_time = False
                else:
                    arr.append(dictionary)
                dictionary = {}
                dictionary[attr] = value
                dictionary['scores'] = {}
            else:
                dictionary['scores'][attr] = round(float(value), 3)
        arr.append(dictionary)
        return arr
            

if __name__ == "__main__":
    baseline = "baseline"
    model = "our_model"
    dataset = "staple"
    metric = 'TER'
    lines = []
    parser = Parser()
    with open(f'{baseline}/{dataset}_result.txt') as file:
        for line in file:
            line = re.sub("\n", "", line)
            lines.append(line)
    baseline_data = parser(lines)
    lines = []
    with open(f'{model}/{dataset}_result.txt') as file:
        for line in file:
            line = re.sub("\n", "", line)
            lines.append(line)
    model_data = parser(lines)
    data_dict = {"": ["Baseline", "LAMPAT (our)"]}
    for d1, d2 in zip(baseline_data, model_data):
        lang = d1['Lang']
        data_dict[lang] = [d1['scores'][metric], d2['scores'][metric]]
    df = pd.DataFrame(data_dict)
    df['Avg'] = [
        np.round(np.mean(df.iloc[0, 1:].values), 3),
        np.round(np.mean(df.iloc[1, 1:].values), 3),
    ]
    df.to_csv(f'{dataset}_{metric}.csv', index=False)