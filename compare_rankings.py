from scipy.stats import kendalltau
import json

FILE1 = 'ranking_LLMs/datasets/datasets--open-llm-leaderboard--results.json'
with open(FILE1, 'r') as f:
    data1 = json.load(f)

FILE2 = 'ranking_LLMs/datasets/melo.json'
with open(FILE2, 'r') as f:
    data2 = json.load(f)

melo_ranks = [i for i, _ in enumerate(data2['models'])]
ids = {m[0]: i for i, m in enumerate(data2['models'])}

data1 = sorted(data1, key=lambda x: x['all'], reverse=True)
naive_ranks = [ids[m['model']] for m in data1 if m['model'] in ids]

print(kendalltau(melo_ranks, naive_ranks))
