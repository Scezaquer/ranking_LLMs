import nashpy
import json
import numpy as np
from poprank.rates import MeloRate
from copy import deepcopy


def load_data(file, cutoff):
    with open(file, 'r') as f:
        data = json.load(f)

    models = []
    tests = [
        "harness|arc:challenge|25",
        "harness|hellaswag|10",
        "harness|truthfulqa:mc|0",
        "all",
        "harness|drop|3",
        "harness|gsm8k|5",
        "harness|winogrande|5",
        "hendrycksTest"
    ]

    if cutoff:
        data = data[:cutoff]

    for x in data:
        models.append(x['model'])

    return data, models, tests


def compute_nash(data, models, tests):
    empirical_payoff_matrix = np.zeros((len(models), len(tests)))

    for i, x in enumerate(data):
        for j, y in enumerate(tests):
            if y in x and x[y] is not None and not np.isnan(x[y]):
                val = x[y]
            else:
                val = .5  # Fill in non-existing values
            empirical_payoff_matrix[i, j] = val

    # empirical_payoff_matrix = np.round(empirical_payoff_matrix - .5, 7)

    empirical_game = nashpy.Game(empirical_payoff_matrix)
    nashs = empirical_game.vertex_enumeration()

    # nashs = [nash for nash in nashs]

    n = nashs.__next__()

    save = {
        "models": {m: n for m, n in zip(models, n[0])},
        "tests": {t: n for t, n in zip(tests, n[1])}
    }

    with open("ranking_LLMs/datasets/nash_avg.json", "w") as f:
        json.dump(save, f)


# -------- Melo ---------- #
def compute_melo(data, models, tests):

    model_perf_dict = {
        m["model"]: {t: m[t] if t in m else None for t in tests} for m in data}
    avg = dict()

    for x in model_perf_dict:
        for y in model_perf_dict[x]:
            if model_perf_dict[x][y] is not None:
                if y in avg:
                    avg[y][0] += model_perf_dict[x][y]
                    avg[y][1] += 1
                else:
                    avg[y] = [model_perf_dict[x][y], 1]

    total_avg = sum([i*j for i, j in avg.values()]) / \
        sum([j for _, j in avg.values()])
    for x in model_perf_dict:
        for y in model_perf_dict[x]:
            if model_perf_dict[x][y] is not None:
                model_perf_dict[x][y] -= total_avg
            else:
                model_perf_dict[x][y] = 0  # avg[y][0]/avg[y][1]

    k = 2
    lr1 = 0.0001
    lr2 = 0.01

    def _build_omega(k):
        omega = np.zeros((2*k, 2*k))
        e = np.atleast_3d(np.identity(2*k))
        for i in range(k):
            omega += e[:, 2*i] @ e[:, 2*i+1].T - e[2*i+1] @ e[2*i].T
        return omega

    def _sigmoid(x):
        return 1/(1+np.exp(-x))

    elos = [MeloRate(0, 0, k=k) for x in models]
    task_difficulty = [MeloRate(0, 0, k=k) for x in tests]

    new_elos = deepcopy(elos)
    new_task_difficulty = deepcopy(task_difficulty)

    # Outcomes must be in interval [0, 1]
    # k_factor must be positive int

    # Perhaps decompose an observed WIN/LOSS matrix in to a C Omega C'
    # for better initial params?

    # Initialize C matrix
    u_matrix = np.array([e.vector for e in elos])
    v_matrix = np.array([e.vector for e in task_difficulty])

    omega = _build_omega(k)

    for k in range(10):
        for i, m in enumerate(models):
            if not i % 100:
                print(i, end="\r")
            for j, t in enumerate(tests):
                p0_id = i
                p1_id = j
                rating0 = new_elos[p0_id].mu
                rating1 = new_task_difficulty[p1_id].mu

                adjustment_matrix = u_matrix @ omega @ v_matrix.T
                p1_adjustment = adjustment_matrix[p0_id, p1_id]

                # Expected win probability
                win_prob = _sigmoid(rating0 - rating1 + p1_adjustment)

                # Delta between expected and actual win
                delta = model_perf_dict[m][t] - win_prob

                # Update ratings. r has higher lr than c
                new_elos[p0_id].mu += lr1*delta
                new_task_difficulty[p1_id].mu -= lr1*delta

                tmp_u_mat = np.array(u_matrix)
                tmp_v_mat = np.array(v_matrix)

                tmp_u_mat[p0_id] = \
                    u_matrix[p0_id] + lr2 * delta * (omega @ v_matrix[p1_id]).T
                tmp_u_mat[p1_id] = \
                    u_matrix[p1_id] - lr2 * delta * (omega @ u_matrix[p0_id]).T

                u_matrix = tmp_u_mat
                v_matrix = tmp_v_mat

                new_elos[p0_id].vector = list(u_matrix[p0_id])
                new_task_difficulty[p1_id].vector = list(v_matrix[p1_id])

    new_elos = [x.mu for x in new_elos]
    task_difficulty = [x.mu for x in task_difficulty]

    save = {
        "models": sorted(
            zip(models, new_elos), key=lambda x: x[1], reverse=True),
        "tests": sorted(
            zip(tests, task_difficulty), key=lambda x: x[1], reverse=True)
    }

    with open("ranking_LLMs/datasets/melo.json", "w") as f:
        json.dump(save, f)


cutoff = 0
file = 'ranking_LLMs/datasets/datasets--open-llm-leaderboard--results.json'
data, models, tests = load_data(file, cutoff)
compute_nash(data, models, tests)
compute_melo(data, models, tests)
