import os
import re
import pandas as pd

if __name__ == "__main__":
    # input only
    path = "eval_dataset/wmt19_input"
    langs = os.listdir(path)
    for lang in langs:
        sents = []
        filename = f'{path}/{lang}/validation.txt'
        with open(filename, 'r') as file:
            for line in file:
                sents.append(re.sub("\n", "", line))

        df = pd.DataFrame({"input": sents, "prediction": sents})
        df.to_csv(f'{path}/{lang}/result.csv', index=False)

    # one reference
    # opusparcus_input & pawsx_input
    path = "eval_dataset/opusparcus_input"
    langs = os.listdir(path)
    for lang in langs:
        filename = f'{path}/{lang}/test.csv'
        df = pd.read_csv(filename)
        new_df = pd.DataFrame()
        new_df['prediction'] = df['input'].values
        new_df['reference'] = df['target'].values
        new_df['input'] = df['input'].values
        new_df.to_csv(f'{path}/{lang}/result.csv', index=False)

    # path = "eval_dataset/pawsx_input"
    langs = os.listdir(path)
    for lang in langs:
        filename = f'{path}/{lang}/test.csv'
        df = pd.read_csv(filename)
        new_df = pd.DataFrame()
        new_df['prediction'] = df['sentence1'].values
        new_df['reference'] = df['sentence2'].values
        new_df['input'] = df['sentence1'].values
        new_df.to_csv(f'{path}/{lang}/result.csv', index=False)

    # multi reference
    # staple_input
    path = "eval_dataset/staple_input"
    langs = os.listdir(path)
    for lang in langs:
        filename = f'{path}/{lang}/test.csv'
        df = pd.read_csv(filename)
        new_df = pd.DataFrame()
        new_df['input'] = df['Input'].values
        new_df['prediction'] = df['Input'].values
        new_df['reference1'] = df['Sent1'].values
        new_df['reference2'] = df['Sent2'].values
        new_df['reference3'] = df['Sent3'].values
        new_df['reference4'] = df['Sent4'].values
        new_df.to_csv(f'{path}/{lang}/result.csv', index=False)