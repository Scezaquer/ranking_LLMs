import os
import json
import numpy as np

FILE = (f"{os.path.dirname(__file__)}/datasets/datasets--open-llm-leaderboard-"
        "-results/snapshots/a93bb511120ff2a762b05391cbe693fc037e3db6")

reformatted = []

for x in os.listdir(FILE):
    for y in os.listdir(os.path.join(FILE, x)):

        results = {}
        try:
            for z in os.listdir(os.path.join(FILE, x, y)):
                with open(os.path.join(FILE, x, y, z), 'r') as f:
                    raw = json.load(f)
                    results.update(raw['results'])
        except FileNotFoundError:
            print(f"couldn't open {os.path.join(x, y)}")
            continue

        tmp = {"model": os.path.join(x, y)}
        for a in results:
            if "acc_norm" in results[a]:
                tmp[a] = results[a]["acc_norm"]
            elif "acc" in results[a]:
                tmp[a] = results[a]["acc"]
            elif "f1" in results[a]:
                tmp[a] = results[a]["f1"]
            else:
                tmp[a] = results[a]["mc2"]
            if np.isnan(tmp[a]):
                tmp[a] = None
        s = 0
        n = 0
        for a in tmp:
            if "hendrycksTest" in a:
                s += tmp[a]
                n += 1
        tmp["hendrycksTest"] = s / n if n != 0 else None
        reformatted.append(tmp)

json_object = json.dumps(reformatted, indent=4)
FILE = 'ranking_LLMs/datasets/datasets--open-llm-leaderboard--results.json'
with open(FILE, 'w') as f:
    f.write(json_object)
