import re

inputs = []
preds = []
refs = []

with open('data/trinh_gan_vae.txt') as file:
    for i, line in enumerate(file):
        line = re.sub('\n', '', line)
        if i % 4 == 0:
            # input
            inp = line[5:]
            inputs.append(inp)
        elif i % 4 == 1:
            # predict
            pred = re.sub('<eos>', '', line[5:])
            preds.append(pred)
        elif i % 4 == 2:
            # target
            tar = re.sub('<eos>', '', line[5:])
            refs.append(tar)
print(len(inputs))
print(len(preds))
print(len(refs))
inputs = inputs[:-1]
import pandas as pd
df = pd.DataFrame({'Input': inputs, 'Predict': preds, 'Reference': refs})
df.to_csv('data/gan_vae.csv', index=False)

