import pandas as pd
import json

df = pd.read_parquet("ranking_LLMs/datasets/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet")

reformatted = []

win_val = set()

for i, row in df.iterrows():
    reformatted.append({x: row[x] for x in df.columns})
    reformatted[-1]['conversation_a'] = reformatted[-1]['conversation_a'].tolist()
    reformatted[-1]['conversation_b'] = reformatted[-1]['conversation_b'].tolist()
    win_val.add(reformatted[-1]['winner'])

# Serializing json
json_object = json.dumps(reformatted, indent=4)

# Writing to sample.json
with open("ranking_LLMs/datasets/gpt4_pair-00000-of-00001-c0b431264a82ddc0.json", "w") as f:
    f.write(json_object)
#reformatted = df.to_json("ranking_LLMs/train-00000-of-00001-cced8514c7ed782a.json")