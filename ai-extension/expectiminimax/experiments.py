import pyspiel
from expectiminimax.algorithms import get_best_action
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import Counter


def run_and_plot_leduc(depth, heuristic_fn, heuristic_name=None):
    """
    Runs expectiminimax best‐action search over all private‐card pairs
    for player 0 at the given depth & heuristic, then plots the results.
    """
    game = pyspiel.load_game("leduc_poker")
    agent = 0
    act_count = len(game.new_initial_state().legal_actions())

    results = []
    for i in range(act_count - 1):
        for j in range(i + 1, act_count):
            for a, b in [(i, j), (j, i)]:
                state = game.new_initial_state()
                state.apply_action(a)
                state.apply_action(b)
                score, action = get_best_action(state, depth, agent, heuristic_fn)
                action_str = state.action_to_string(agent, action)
                results.append((a, b, score, action_str))

    # unpack
    xs = [r[0] for r in results]
    ys = [r[1] for r in results]
    scores = [r[2] for r in results]
    acts = [r[3] for r in results]

    marker_map = {"call": "o", "raise": "^", "fold": "s"}

    plt.figure(figsize=(8, 6))
    for x, y, sc, a in zip(xs, ys, scores, acts):
        plt.scatter(
            x,
            y,
            s=100,
            c=[sc],
            cmap="viridis",
            vmin=min(scores),
            vmax=max(scores),
            marker=marker_map.get(a, "o"),
            edgecolor="k",
            linewidth=0.5,
        )

    cb = plt.colorbar()
    cb.set_label("Expectiminimax Score", rotation=270, labelpad=15)

    plt.xlabel("Player 0 Card Index")
    plt.ylabel("Player 1 Card Index")
    title_h = heuristic_name or heuristic_fn.__name__
    plt.title(f"Leduc Poker Best‐Action Scores\n(depth={depth}, heuristic={title_h})")

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


def run_limit(start_state, depth, heuristic_fn, heuristic_name, k_samples, trials):
    """
    Run expectiminimax multiple times on fresh clones of start_state,
    average the heuristic scores (to smooth out sampling noise),
    and report the most common action seen.
    """
    scores = []
    actions = []
    for t in range(trials):
        state = start_state.clone()
        agent = 0  # can parameterize if we like
        score, action = get_best_action(state, depth, agent, heuristic_fn, k_samples)
        scores.append(score)
        actions.append(action)

    avg_score = sum(scores) / trials
    action_counts = Counter(actions)
    most_common_action, freq = action_counts.most_common(1)[0]
    # (for sanity we also fetch its string)
    action_str = start_state.action_to_string(agent, most_common_action)

    print(f"[{heuristic_name}] depth={depth}  k={k_samples}  trials={trials}")
    print(f"    scores: {['{:.3f}'.format(s) for s in scores]}")
    print(f"    avg   : {avg_score:.3f}")
    print(
        f"    best–action (mode {freq}/{trials}): "
        f"{most_common_action} ({action_str})\n"
    )

    return avg_score, most_common_action
