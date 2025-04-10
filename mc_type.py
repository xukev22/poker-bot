# How does every visit mc compare against first visit mc?
import os
import statistics
import numpy as np
import rlcard
from agents import FirstVisitMCAgent, RandomAgent, HumanAgent, EveryVisitMCAgent
from experiments import play_episodes, evaluate_agents
from utils import (
    process_leduc_state_v1,
    process_leduc_state_v2,
    process_leduc_state_v3,
    process_leduc_state_v4,
)

import matplotlib.pyplot as plt

# config
NUM_EPISODES = 10000
UPDATE_FREQ = 50
state_processors = [
    process_leduc_state_v1,
    process_leduc_state_v2,
    process_leduc_state_v3,
    process_leduc_state_v4,
]
epsilons = [0.01, 0.1]
gammas = [1, 0.99, 0.9, 0.5]

env = rlcard.make("leduc-holdem")
random_agent = RandomAgent()

# loop over all combinations of config
for sp in state_processors:
    for e in epsilons:
        for g in gammas:
            print(
                f"\n=== Starting trials for EveryVisit vs. FirstVisit "
                f"w/ e={e} gamma={g} using state processor {sp.__name__} ==="
            )

            # create new agents for each set of hyperparameters
            agent_e = EveryVisitMCAgent(epsilon=e, gamma=g)
            agent_f = FirstVisitMCAgent(epsilon=e, gamma=g)

            # train them against each other on the given state
            _ = play_episodes(
                env,
                agent_e,
                agent_f,
                num_episodes=NUM_EPISODES,
                do_update=True,
                update_freq=UPDATE_FREQ,
                state_transformer=sp,
            )

            # eval agent_e vs random for 10000 episodes
            e_vs_rand_payoffs = play_episodes(
                env, agent_e, random_agent, 10000, do_update=False
            )
            # agent_e is player 0, so payoff is at index [0]
            e_rewards = [p[0] for p in e_vs_rand_payoffs]
            e_cumulative = np.cumsum(e_rewards)

            # Evaluate agent_f vs random for 10000 episodes
            f_vs_rand_payoffs = play_episodes(
                env, random_agent, agent_f, 10000, do_update=False
            )
            # agent_f is player 1, so payoff is at index 1
            f_rewards = [p[1] for p in f_vs_rand_payoffs]
            f_cumulative = np.cumsum(f_rewards)

            # Plot both lines in a single figure
            plt.figure()
            plt.plot(e_cumulative, label="EveryVisit MC vs Random")
            plt.plot(f_cumulative, label="FirstVisit MC vs Random")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title(f"Cumulative Rewards ({sp.__name__}) e={e}, gamma={g}")
            plt.legend()

            # Save plot to file
            filename = f"graphs/EvF_comparison_{sp.__name__}_e{e}_g{g}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close()
            print(f"Saved combined plot to {filename}")

            e_scores = []
            f_scores = []
            p0_avg, p1_avg = evaluate_agents(
                env, agent_e, agent_f, num_episodes=1000, plot=False
            )
            e_scores.append(p0_avg)  # agent_e average payoff
            f_scores.append(p1_avg)  # agent_f average payoff

            mean_e = statistics.mean(e_scores)
            mean_f = statistics.mean(f_scores)
            print(f"Over 1000 episodes of EveryVisit vs. FirstVisit:")
            print(f"  EveryVisit average payoff = {mean_e}")
            print(f"  FirstVisit average payoff = {mean_f}")
            if mean_e > mean_f:
                print("  => EveryVisitMCAgent performs better.")
            elif mean_f > mean_e:
                print("  => FirstVisitMCAgent performs better.")
            else:
                print("  => They perform equally.")
