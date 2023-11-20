import json
from os.path import dirname

import numpy as np
import random
import plotly.express as px

from popcore import Interaction
from poprank.rates import EloRate, Glicko2Rate, GlickoRate, Rate, TrueSkillRate, MeloRate
from poprank.functional import (bayeselo, elo, windrawlose, glicko, glicko2,
                                trueskill, mElo)


def str_to_outcome(s: str):
    """Turn the string describing the outcome into chess notation"""
    if s == "model_a":
        return (1, 0)
    if s == "model_b":
        return (0, 1)
    return (0.5, 0.5)


# Import tournament dataset.
# Originally from https://lmsys.org/blog/2023-05-03-arena/

path = f"{dirname(__file__)}/clean_battle_20230717.json"
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

for match in file:
    # Add the players to the list of contenders if they aren't in already
    players.add(match['model_a'])
    players.add(match['model_b'])

    # Turn the outcome to chess notation
    outcome = str_to_outcome(match['winner'])

    # Store the interaction
    interac = Interaction([match['model_a'], match['model_b']], outcome)
    interactions.append(interac)

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
# best lr so far: 0.0001, 0.01
melos2 = mElo(players, interactions, melos2, k=1, iterations=1, lr1=0.0001, lr2=0.01)
melos4 = mElo(players, interactions, melos4, k=2, iterations=1, lr1=0.0001, lr2=0.01)
melos10 = mElo(players, interactions, melos10, k=5, iterations=1, lr1=0.0001, lr2=0.01)
melos20 = mElo(players, interactions, melos20, k=10, iterations=1, lr1=0.0001, lr2=0.01)
"""nashavg = nash_avg(players, interactions)"""


sequential_elo = [EloRate(0) for x in players]
for match in interactions:
    sequential_elo = elo(players, [match], sequential_elo, k_factor=4)

# Rank the players based on their bayeselo ratings
players, elos, bayeselos, selo, glickos, glickos2, trueskills, wdl, wins, draws, losses, played, melos2, melos4, melos10, melos20= \
    [list(t) for t in zip(*sorted(zip(players, elos, bayeselos, sequential_elo, glickos, glickos2, trueskills, wdl, wins, draws, losses, played, melos2, melos4, melos10, melos20), key=lambda x: x[8].mu/x[11].mu, reverse=True))]

# Print the results
for p, e, b, se, g, g2, t, wdl_, w, d, l, pl, ml2, ml4, ml10, ml20 in zip(players, elos, bayeselos, selo, glickos, glickos2, trueskills, wdl, wins, draws, losses, played, melos2, melos4, melos10, melos20):
    print(f"| {p} | {round(b.mu, 1)} | {round(e.mu, 1)} | {round(se.mu, 1)} | {round(g.mu, 1)} | {round(g2.mu, 1)} | {round(t.mu, 1)} | {round(ml2.mu, 5)}  | {round(ml4.mu, 5)}  | {round(ml10.mu, 5)}  | {round(ml20.mu, 5)} | {round(w.mu/pl.mu*100, 1)} | {round(d.mu/pl.mu*100, 1)} | {round(l.mu/pl.mu*100, 1)} | {wdl_.mu} | {w.mu} | {d.mu} | {l.mu} | {pl.mu}")


"""for p, n in zip(players, nashavg):
    print(p, n.mu)"""

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
        win_matrix[player_index[match['model_a']]][player_index[match['model_b']]] += 1
        loss_matrix[player_index[match['model_b']]][player_index[match['model_a']]] += 1
    elif match['winner'] == 'model_b':
        win_matrix[player_index[match['model_b']]][player_index[match['model_a']]] += 1
        loss_matrix[player_index[match['model_a']]][player_index[match['model_b']]] += 1
    else:
        draw_matrix[player_index[match['model_b']]][player_index[match['model_a']]] += 1
        draw_matrix[player_index[match['model_a']]][player_index[match['model_b']]] += 1
    played_matrix[player_index[match['model_b']]][player_index[match['model_a']]] += 1
    played_matrix[player_index[match['model_a']]][player_index[match['model_b']]] += 1

winrate_matrix = np.divide(win_matrix + 1/2*draw_matrix, played_matrix)*100

elo_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(selo):
    for j, y in enumerate(selo):
        elo_prediction_matrix[i][j] = x.expected_outcome(y)*100

bayeselo_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(bayeselos):
    for j, y in enumerate(bayeselos):
        bayeselo_prediction_matrix[i][j] = x.expected_outcome(y)*100

glicko_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(glickos):
    for j, y in enumerate(glickos):
        glicko_prediction_matrix[i][j] = x.expected_outcome(y)*100

glicko2_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(glickos2):
    for j, y in enumerate(glickos2):
        glicko2_prediction_matrix[i][j] = x.expected_outcome(y)*100

trueskill_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(trueskills):
    for j, y in enumerate(trueskills):
        trueskill_prediction_matrix[i][j] = x.expected_outcome(y)*100

melo2_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(melos2):
    for j, y in enumerate(melos2):
        melo2_prediction_matrix[i][j] = x.expected_outcome(y)*100

melo4_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(melos4):
    for j, y in enumerate(melos4):
        melo4_prediction_matrix[i][j] = x.expected_outcome(y)*100

melo10_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(melos10):
    for j, y in enumerate(melos10):
        melo10_prediction_matrix[i][j] = x.expected_outcome(y)*100

melo20_prediction_matrix = np.zeros((len(players), len(players)))
for i, x in enumerate(melos20):
    for j, y in enumerate(melos20):
        melo20_prediction_matrix[i][j] = x.expected_outcome(y)*100

print("True winrate")
print(np.round(winrate_matrix, 1))

print("Elo predicted winrate")
print(np.round(elo_prediction_matrix, 1))

print("L2 distances")
print(f"| Sequential Elo | {round(np.linalg.norm(winrate_matrix-elo_prediction_matrix), 1)} |")
print(f"| Bayeselo | {round(np.linalg.norm(winrate_matrix-bayeselo_prediction_matrix), 1)} |")
print(f"| Glicko | {round(np.linalg.norm(winrate_matrix-glicko_prediction_matrix), 1)} |")
print(f"| Glicko2 | {round(np.linalg.norm(winrate_matrix-glicko2_prediction_matrix), 1)} |")
print(f"| Trueskill | {round(np.linalg.norm(winrate_matrix-trueskill_prediction_matrix), 1)} |")
print(f"| Melo2 | {round(np.linalg.norm(winrate_matrix-melo2_prediction_matrix), 1)} |")
print(f"| Melo4 | {round(np.linalg.norm(winrate_matrix-melo4_prediction_matrix), 1)} |")
print(f"| Melo10 | {round(np.linalg.norm(winrate_matrix-melo10_prediction_matrix), 1)} |")
print(f"| Melo20 | {round(np.linalg.norm(winrate_matrix-melo20_prediction_matrix), 1)} |")


def visualize_winrate(matrix, title, filename):
    fig = px.imshow(matrix, x=players, y=players, color_continuous_scale='RdBu',
                    text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B: Loser",
                      yaxis_title="Model A: Winner",
                      xaxis_side="top", height=600, width=600,
                      title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>")

    fig.write_image(f"ranking_LLMs/figures/{filename}.png")


visualize_winrate(winrate_matrix, title="True winrates on test set", filename="true_winrates")
visualize_winrate(elo_prediction_matrix, title="Winrates predicted by elo on test set", filename="elo_prediction")
visualize_winrate(bayeselo_prediction_matrix, title="Winrates predicted by Bayeselo on test set", filename="bayeselo_prediction")
visualize_winrate(glicko_prediction_matrix, title="Winrates predicted by Glicko on test set", filename="glicko_prediction")
visualize_winrate(glicko2_prediction_matrix, title="Winrates predicted by Glicko2 on test set", filename="glicko2_prediction")
visualize_winrate(trueskill_prediction_matrix, title="Winrates predicted by TrueSkill on test set", filename="trueskill_prediction")
visualize_winrate(melo2_prediction_matrix, title="Winrates predicted by mElo2 on test set", filename="melo2_prediction")
visualize_winrate(melo4_prediction_matrix, title="Winrates predicted by mElo4 on test set", filename="melo4_prediction")
visualize_winrate(melo10_prediction_matrix, title="Winrates predicted by mElo10 on test set", filename="melo10_prediction")
visualize_winrate(melo20_prediction_matrix, title="Winrates predicted by mElo20 on test set", filename="melo20_prediction")