# File to reformat datasets to be able to use them in ranking_llms.py

import pandas as pd
import json

f = "ranking_LLMs/datasets/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet"
df = pd.read_parquet(f)

reformat = []

win_val = set()

for i, row in df.iterrows():
    reformat.append({x: row[x] for x in df.columns})
    reformat[-1]['conversation_a'] = reformat[-1]['conversation_a'].tolist()
    reformat[-1]['conversation_b'] = reformat[-1]['conversation_b'].tolist()
    win_val.add(reformat[-1]['winner'])

json_object = json.dumps(reformat, indent=4)
nf = "ranking_LLMs/datasets/gpt4_pair-00000-of-00001-c0b431264a82ddc0.json"
with open(nf, "w") as f:
    f.write(json_object)
