import pyspiel
from expectiminimax.algorithms import get_best_action
from expectiminimax.heuristics import h_perfect_info
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load the game and initial state
game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
agent = 0
# max should not be more than 10 in 2bet max like leduc
# (call raise raise call, card, check raise raise call)
depth = 10

act_count = len(state.legal_actions())

results = []

# deal first two cards (every combination)
for i in range(act_count - 1):
    for j in range(i + 1, act_count):
        state = game.new_initial_state()

        state.apply_action(i)
        state.apply_action(j)
        # compute best action
        score, action = get_best_action(state, depth, agent, h_perfect_info)
        action_str = state.action_to_string(agent, action)
        print(
            f"Using heuristic: {h_perfect_info} Card {i} vs. Card {j} | Best action found at depth {depth}: {action_str} (score: {score:.3f})"
        )

        results.append((i, j, score, action_str))

        state = game.new_initial_state()

        state.apply_action(j)
        state.apply_action(i)
        # compute best action
        score, action = get_best_action(state, depth, agent, h_perfect_info)
        action_str = state.action_to_string(agent, action)
        print(
            f"Card {j} vs. Card {i} Best action at depth {depth}: {action_str} (score: {score:.3f})"
        )

        results.append((j, i, score, action_str))


# ── unpack results: a list of (i, j, score, action_str)
# results = [(0,1,0.52,'call'), (0,2,0.35,'fold'), …]

xs = [r[0] for r in results]
ys = [r[1] for r in results]
scores = [r[2] for r in results]
acts = [r[3] for r in results]

# map actions to marker styles
marker_map = {"call": "o", "raise": "^", "fold": "s"}

plt.figure(figsize=(8, 6))
for x, y, sc, a in zip(xs, ys, scores, acts):
    plt.scatter(
        x,
        y,
        s=100,
        c=[sc],  # single‐value list so each point gets its own color
        cmap="viridis",
        vmin=min(scores),
        vmax=max(scores),
        marker=marker_map.get(a, "o"),
        edgecolor="k",
        linewidth=0.5,
    )

# colorbar for the score
cb = plt.colorbar()
cb.set_label("Expectiminimax Score", rotation=270, labelpad=15)

plt.xlabel("Player 0 Card Index")
plt.ylabel("Player 1 Card Index")
plt.title(f"Leduc Poker Best‐Action Scores (depth={depth})")

# build a custom legend for actions
legend_handles = [
    mlines.Line2D(
        [], [], color="k", marker=m, linestyle="None", markersize=8, label=a.title()
    )
    for a, m in marker_map.items()
]
plt.legend(handles=legend_handles, title="Action", loc="upper left")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# print("Initial state:")
# print(state)

# print(state.is_chance_node())
# print(state.legal_actions())

# # J1
# state.apply_action(0)
# # K2
# state.apply_action(5)
