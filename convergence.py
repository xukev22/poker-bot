import os
import numpy as np
import matplotlib.pyplot as plt
import rlcard
from agents import FirstVisitMCAgent, EveryVisitMCAgent, RandomAgent
from experiments import play_episodes
from utils import process_leduc_state_v1

# Set parameters
training_points = [1, 5, 10, 50, 100, 1000, 5000, 10000, 100000, 1000000]
num_eval_episodes = 10000  # Evaluation episodes against Random Agent
epsilon = 0.01
gamma = 0.9
state_processor = process_leduc_state_v1

# Storage for cumulative rewards for each algorithm
everyvisit_results = []
firstvisit_results = []

env = rlcard.make("leduc-holdem")

# For reproducibility you could set seeds here if desired

# Iterate over each training milestone
for train_eps in training_points:
    print(f"\n--- Training for {train_eps} episodes ---")
    # Reinitialize agents (so each milestone has a clean slate)
    agent_ev = EveryVisitMCAgent(
        epsilon=epsilon, gamma=gamma, state_transformer=state_processor
    )
    agent_fv = FirstVisitMCAgent(
        epsilon=epsilon, gamma=gamma, state_transformer=state_processor
    )
    random_agent = RandomAgent()

    # Train agents against each other (using orientation 1: EV as player 0, FV as player 1)
    play_episodes(
        env,
        agent_ev,  # Player 0
        agent_fv,  # Player 1
        num_episodes=train_eps,
        do_update=True,
        update_freq=100,
        state_transformer=state_processor,
    )

    # Evaluate EveryVisit agent: let it be Player 0 vs Random Agent (Player 1)
    ev_vs_rand_payoffs = play_episodes(
        env, agent_ev, random_agent, num_eval_episodes, do_update=False
    )
    # Sum up rewards for Player 0 (EveryVisit)
    ev_cum_reward = np.sum([p[0] for p in ev_vs_rand_payoffs])

    # Evaluate FirstVisit agent: let it be Player 1 vs Random Agent (Player 0)
    fv_vs_rand_payoffs = play_episodes(
        env, random_agent, agent_fv, num_eval_episodes, do_update=False
    )
    # Sum up rewards for Player 1 (FirstVisit)
    fv_cum_reward = np.sum([p[1] for p in fv_vs_rand_payoffs])

    everyvisit_results.append(ev_cum_reward)
    firstvisit_results.append(fv_cum_reward)

    print(f"After {train_eps} episodes:")
    print(f"  EveryVisit cumulative reward vs Random: {ev_cum_reward}")
    print(f"  FirstVisit cumulative reward vs Random: {fv_cum_reward}")

# Plotting the convergence bar chart
x = np.arange(len(training_points))  # group positions
width = 0.35  # width of each bar

fig, ax = plt.subplots()
bars_ev = ax.bar(x - width / 2, everyvisit_results, width, label="EveryVisit")
bars_fv = ax.bar(x + width / 2, firstvisit_results, width, label="FirstVisit")

ax.set_xlabel("Number of Training Episodes")
ax.set_ylabel("Cumulative Reward (vs. Random Agent)")
ax.set_title("Convergence Comparison: EveryVisit vs FirstVisit Monte Carlo Agents")
ax.set_xticks(x)
ax.set_xticklabels([str(tp) for tp in training_points])
ax.legend()

# Optionally annotate bars with their values for clarity
for bar in bars_ev + bars_fv:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # offset in points
        textcoords="offset points",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
# Save or show the figure
save_path = "graphs/convergence_comparison.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()

print(f"Saved convergence comparison graph to {save_path}")
