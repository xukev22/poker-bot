import pyspiel
from expectiminimax.algorithms import get_best_action
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from collections import Counter


def run_and_plot_leduc(
    depth, heuristic_fn, heuristic_name=None, out_dir="../ai-extension/graphs"
):
    """
    Runs expectiminimax best‐action search over all private‐card pairs
    for player 0 at the given depth & heuristic, then plots the results
    *and saves the figure* to out_dir.
    """
    # ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)

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

    # build a filename
    safe_name = title_h.replace(" ", "_")
    fname = f"leduc_d{depth}_{safe_name}.png"
    path = os.path.join(out_dir, fname)

    # save to disk
    plt.savefig(path, dpi=300)
    print(f"Saved plot to {path}")

    # optionally still show it
    plt.show()


import os
import matplotlib.pyplot as plt
from collections import Counter


def run_and_plot_limit(
    start_state,
    depth,
    heuristic_fn,
    heuristic_name,
    k_samples,
    trials,
    out_dir="../ai-extension/graphs",
):
    """
    Run expectiminimax multiple times on fresh clones of start_state,
    average the heuristic scores (to smooth out sampling noise),
    and report the most common action seen—but also
    save two plots: a histogram of scores and a bar chart of action counts.
    """
    os.makedirs(out_dir, exist_ok=True)

    scores = []
    actions = []
    agent = 0

    for t in range(trials):
        state = start_state.clone()
        score, action = get_best_action(state, depth, agent, heuristic_fn, k_samples)
        scores.append(score)
        actions.append(action)

    avg_score = sum(scores) / trials
    action_counts = Counter(actions)
    most_action, freq = action_counts.most_common(1)[0]
    most_str = start_state.action_to_string(agent, most_action)

    # build filename base
    safe_h = heuristic_name.replace(" ", "_")
    base = f"limit_d{depth}_k{k_samples}_t{trials}_{safe_h}"
    png_path = os.path.join(out_dir, base + ".png")

    # make figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1) Histogram of scores
    ax1.hist(scores, bins="auto", edgecolor="black")
    ax1.set_title("Score Distribution")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Frequency")

    # 2) Bar chart of action frequencies
    labels = [start_state.action_to_string(agent, a) for a in action_counts.keys()]
    counts = list(action_counts.values())
    ax2.bar(labels, counts, edgecolor="black")
    ax2.set_title("Action Counts")
    ax2.set_ylabel("Trials")

    fig.suptitle(
        f"{heuristic_name} | depth={depth}, k={k_samples}, trials={trials}\n"
        f"avg_score={avg_score:.3f}, mode={most_str} ({freq}/{trials})"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and show
    plt.savefig(png_path, dpi=300)
    print(f"→ Saved plots to {png_path}")
    plt.show()

    return avg_score, most_action
