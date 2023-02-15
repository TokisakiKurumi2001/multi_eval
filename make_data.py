import pandas as pd
import re

output1 = []
output2 = []
output3 = []
output4 = []
truth = []

with open('data/output_corrupted.txt') as file:
    cnt = 1
    for line in file:
        line = re.sub('\n', '', line)
        if cnt % 5 == 1:
            output1.append(line)
        elif cnt % 5 == 2:
            output2.append(line)
        elif cnt % 5 == 3:
            output3.append(line)
        elif cnt % 5 == 4:
            output4.append(line)
        else:
            truth.append(line)
        cnt += 1

df = pd.DataFrame({"Output1": output1, "Output2": output2, "Output3": output3, "Output4": output4, "Truth": truth})
df.to_csv('data/output_corrupted.csv', index=False)