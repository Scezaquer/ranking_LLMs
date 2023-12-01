import json
from os.path import dirname

import numpy as np
import random
import plotly.express as px

from popcore import Interaction
from poprank.rates import (
    EloRate, Glicko2Rate, GlickoRate, Rate, TrueSkillRate, MeloRate)
from poprank.functional import (
    bayeselo, elo, windrawlose, glicko, glicko2, trueskill, mElo)


def str_to_outcome(s: str):
    """Turn the string describing the outcome into chess notation"""
    if s == "model_a":
        return (1, 0)
    if s == "model_b":
        return (0, 1)
    return (0.5, 0.5)


def str_to_zero_sum(s: str):
    """Turn the string describing the outcome into a zero-sum interac"""
    if s == "model_a":
        return (1, -1)
    if s == "model_b":
        return (-1, 1)
    return (0, 0)


# Import tournament dataset.
# Originally from https://lmsys.org/blog/2023-05-03-arena/

FILENAME = "gpt4_pair-00000-of-00001-c0b431264a82ddc0.json"

path = f"{dirname(__file__)}/datasets/{FILENAME}"

with open(path, 'r') as f:
    file = json.load(f)

print(f"Loaded {len(file)} matches")

random.seed(0)

# Separate into training vs testing
random.shuffle(file)
testing_set = file[-round(len(file)/10):]
file = file[:-round(len(file)/10)]

# Get a list of players and interactions
players = set()
interactions = []
zero_sum_interactions = []

for match in file:
    # Add the players to the list of contenders if they aren't in already
    """if match['model_a'] == "oasst-sft-1-pythia-12b"\
       or match['model_b'] == "oasst-sft-1-pythia-12b":
        continue"""

    players.add(match['model_a'])
    players.add(match['model_b'])

    # Turn the outcome to chess notation
    outcome = str_to_outcome(match['winner'])
    zero_sum_outcome = str_to_zero_sum(match['winner'])

    # Store the interaction
    interac = Interaction([match['model_a'], match['model_b']], outcome)
    zero_sum_interac = Interaction(
        [match['model_a'], match['model_b']],
        zero_sum_outcome)
    interactions.append(interac)
    zero_sum_interactions.append(zero_sum_interac)

for match in testing_set:
    players.add(match['model_a'])
    players.add(match['model_b'])

# Initialize the elos to 0
players = list(players)
elos = [EloRate(0) for x in players]
bayeselos = [EloRate(0) for x in players]
glickos = [GlickoRate(0) for x in players]
glickos2 = [Glicko2Rate(0) for x in players]
trueskills = [TrueSkillRate(25, 25/3) for x in players]
wdl = [Rate(0) for x in players]
wins = [Rate(0) for x in players]
draws = [Rate(0) for x in players]
losses = [Rate(0) for x in players]
played = [Rate(0) for x in players]
melos2 = [MeloRate(0, 1, k=1) for x in players]
melos4 = [MeloRate(0, 1, k=2) for x in players]
melos10 = [MeloRate(0, 1, k=5) for x in players]
melos20 = [MeloRate(0, 1, k=10) for x in players]

# Compute the ratings
elos = elo(players, interactions, elos, k_factor=4)
bayeselos = bayeselo(players, interactions, bayeselos)
glickos = glicko(players, interactions, glickos)
glickos2 = glicko2(players, interactions, glickos2)
trueskills = trueskill(players, interactions, trueskills)
wdl = windrawlose(players, interactions, wins, 1, 0, -1)
wins = windrawlose(players, interactions, wins, 1, 0, 0)
draws = windrawlose(players, interactions, draws, 0, 1, 0)
losses = windrawlose(players, interactions, losses, 0, 0, 1)
played = windrawlose(players, interactions, played, 1, 1, 1)
melos2 = mElo(players, interactions, melos2, k=1, lr1=0.0001, lr2=0.01)
melos4 = mElo(players, interactions, melos4, k=2, lr1=0.0001, lr2=0.01)
melos10 = mElo(players, interactions, melos10, k=5, lr1=0.0001, lr2=0.01)
melos20 = mElo(players, interactions, melos20, k=10, lr1=0.0001, lr2=0.01)
# nashavg = nash_avg(players, zero_sum_interactions)


sequential_elo = [EloRate(0) for x in players]
for match in interactions:
    sequential_elo = elo(players, [match], sequential_elo, k_factor=4)

# Rank the players based on their bayeselo ratings
players, elos, bayeselos, selo, glickos, glickos2, trueskills, wdl, wins, \
    draws, losses, played, melos2, melos4, melos10, melos20 = \
    [list(t) for t in zip(*sorted(
        zip(players, elos, bayeselos, sequential_elo, glickos, glickos2,
            trueskills, wdl, wins, draws, losses, played, melos2, melos4,
            melos10, melos20),
        key=lambda x: x[8].mu/x[11].mu if x[11].mu else 0, reverse=True))]

# Print the results
for p, e, b, se, g, g2, t, wdl_, w, d, l, pl, ml2, ml4, ml10, ml20 in\
   zip(players, elos, bayeselos, selo, glickos, glickos2, trueskills, wdl,
       wins, draws, losses, played, melos2, melos4, melos10, melos20):
    print(
        f"| {p} | {round(b.mu, 1)} | {round(e.mu, 1)} | {round(se.mu, 1)} | "
        f"{round(g.mu, 1)} | {round(g2.mu, 1)} | {round(t.mu, 1)} | "
        f"{round(ml2.mu, 5)}  | {round(ml4.mu, 5)}  | {round(ml10.mu, 5)}  |"
        f"{round(ml20.mu, 5)} | {round(w.mu/pl.mu*100 if pl.mu else 0, 1)} |"
        f"{round(d.mu/pl.mu*100 if pl.mu else 0, 1)} | "
        f"{round(l.mu/pl.mu*100 if pl.mu else 0, 1)} | {wdl_.mu} | {w.mu} | "
        f"{d.mu} | {l.mu} | {pl.mu}")

# ---- Predicting winrates ---- #

# We do some lagrange norm to avoid NAN values
player_index = {p: i for i, p in enumerate(players)}
win_matrix = np.zeros((len(players), len(players))) + 1
draw_matrix = np.zeros((len(players), len(players)))
loss_matrix = np.zeros((len(players), len(players))) + 1
played_matrix = np.zeros((len(players), len(players))) + 2

for match in testing_set:
    # Turn the outcome to chess notation
    if match['winner'] == 'model_a':
        p1_id = player_index[match['model_a']]
        p2_id = player_index[match['model_b']]
        win_matrix[p1_id][p2_id] += 1
        loss_matrix[p2_id][p1_id] += 1
    elif match['winner'] == 'model_b':
        win_matrix[p2_id][p1_id] += 1
        loss_matrix[p1_id][p2_id] += 1
    else:
        draw_matrix[p2_id][p1_id] += 1
        draw_matrix[p1_id][p2_id] += 1
    played_matrix[p2_id][p1_id] += 1
    played_matrix[p1_id][p2_id] += 1

winrate_matrix = np.divide(win_matrix + 1/2*draw_matrix, played_matrix)*100


def pred_matrix(ratings, len_players):
    matrix = np.zeros((len_players, len_players))
    for i, x in enumerate(ratings):
        for j, y in enumerate(ratings):
            matrix[i][j] = x.expected_outcome(y)*100
    return matrix


elo_prediction_matrix = pred_matrix(selo, len(players))
bayeselo_prediction_matrix = pred_matrix(bayeselos, len(players))
glicko_prediction_matrix = pred_matrix(glickos, len(players))
glicko2_prediction_matrix = pred_matrix(glickos2, len(players))
trueskill_prediction_matrix = pred_matrix(trueskills, len(players))
melo2_prediction_matrix = pred_matrix(melos2, len(players))
melo4_prediction_matrix = pred_matrix(melos4, len(players))
melo10_prediction_matrix = pred_matrix(melos10, len(players))
melo20_prediction_matrix = pred_matrix(melos20, len(players))

print("True winrate")
print(np.round(winrate_matrix, 1))

print("Elo predicted winrate")
print(np.round(elo_prediction_matrix, 1))

print("L2 distances")
print(
    "| Sequential Elo | "
    f"{round(np.linalg.norm(winrate_matrix-elo_prediction_matrix), 1)} |")
print(
    "| Bayeselo | "
    f"{round(np.linalg.norm(winrate_matrix-bayeselo_prediction_matrix), 1)} |")
print(
    "| Glicko | "
    f"{round(np.linalg.norm(winrate_matrix-glicko_prediction_matrix), 1)} |")
print(
    "| Glicko2 | "
    f"{round(np.linalg.norm(winrate_matrix-glicko2_prediction_matrix), 1)} |")
print(
    "| Trueskill | "
    f"{round(np.linalg.norm(winrate_matrix-trueskill_prediction_matrix), 1)} |"
    )
print(
    "| Melo2 | "
    f"{round(np.linalg.norm(winrate_matrix-melo2_prediction_matrix), 1)} |")
print(
    "| Melo4 | "
    f"{round(np.linalg.norm(winrate_matrix-melo4_prediction_matrix), 1)} |")
print(
    "| Melo10 | "
    f"{round(np.linalg.norm(winrate_matrix-melo10_prediction_matrix), 1)} |")
print(
    "| Melo20 | "
    f"{round(np.linalg.norm(winrate_matrix-melo20_prediction_matrix), 1)} |")


def visualize_winrate(matrix, title, filename):
    fig = px.imshow(
        matrix, x=players, y=players, color_continuous_scale='RdBu',
        text_auto=".2f", title=title)
    fig.update_layout(
        xaxis_title=" Model B: Loser",
        yaxis_title="Model A: Winner",
        xaxis_side="top", height=600, width=600,
        title_y=0.07, title_x=0.5
    )
    fig.update_traces(
        hovertemplate=(
            "Model A: %{y}<br>Model B: %{x}<br>"
            "Fraction of A Wins: %{z}<extra></extra>"))

    fig.write_image(f"ranking_LLMs/figures/{FILENAME[:12]}_{filename}.png")


visualize_winrate(
    winrate_matrix,
    title="True winrates on test set", filename="true_winrates")
visualize_winrate(
    elo_prediction_matrix,
    title="Winrates predicted by elo on test set", filename="elo_prediction")
visualize_winrate(
    bayeselo_prediction_matrix,
    title="Winrates predicted by Bayeselo on test set",
    filename="bayeselo_prediction")
visualize_winrate(
    glicko_prediction_matrix, 
    title="Winrates predicted by Glicko on test set",
    filename="glicko_prediction")
visualize_winrate(
    glicko2_prediction_matrix,
    title="Winrates predicted by Glicko2 on test set",
    filename="glicko2_prediction")
visualize_winrate(
    trueskill_prediction_matrix,
    title="Winrates predicted by TrueSkill on test set",
    filename="trueskill_prediction")
visualize_winrate(
    melo2_prediction_matrix,
    title="Winrates predicted by mElo2 on test set",
    filename="melo2_prediction")
visualize_winrate(
    melo4_prediction_matrix,
    title="Winrates predicted by mElo4 on test set",
    filename="melo4_prediction")
visualize_winrate(
    melo10_prediction_matrix,
    title="Winrates predicted by mElo10 on test set",
    filename="melo10_prediction")
visualize_winrate(
    melo20_prediction_matrix,
    title="Winrates predicted by mElo20 on test set",
    filename="melo20_prediction")