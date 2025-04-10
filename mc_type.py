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
# --------------------
# 1) Original "orientation": agent_e (EveryVisit) is Player 0, agent_f (FirstVisit) is Player 1
# --------------------
for sp in state_processors:
    for e in epsilons:
        for g in gammas:
            print(
                f"\n=== (Orientation 1) EveryVisit vs. FirstVisit "
                f"w/ e={e}, gamma={g}, sp={sp.__name__} ==="
            )

            agent_e = EveryVisitMCAgent(epsilon=e, gamma=g)
            agent_f = FirstVisitMCAgent(epsilon=e, gamma=g)

            # Train them against each other
            play_episodes(
                env,
                agent_e,  # Player 0
                agent_f,  # Player 1
                num_episodes=NUM_EPISODES,
                do_update=True,
                update_freq=UPDATE_FREQ,
                state_transformer=sp,
            )

            # Evaluate agent_e (as Player 0) vs Random
            e_vs_rand_payoffs = play_episodes(
                env, agent_e, random_agent, 10000, do_update=False
            )
            e_rewards = [p[0] for p in e_vs_rand_payoffs]
            e_cumulative = np.cumsum(e_rewards)

            # Evaluate agent_f (as Player 1) vs Random
            # So we keep agent_f in second position to keep it as Player 1
            f_vs_rand_payoffs = play_episodes(
                env, random_agent, agent_f, 10000, do_update=False
            )
            f_rewards = [p[1] for p in f_vs_rand_payoffs]
            f_cumulative = np.cumsum(f_rewards)

            # Plot both lines in a single figure
            plt.figure()
            plt.plot(e_cumulative, label="EveryVisit MC vs Random (Player 0)")
            plt.plot(f_cumulative, label="FirstVisit MC vs Random (Player 1)")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title(f"[Orientation 1] {sp.__name__}, e={e}, gamma={g}")
            plt.legend()

            # Save plot to file
            filename = f"graphs/orient1_EvF_{sp.__name__}_e{e}_g{g}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close()
            print(f"Saved combined plot to {filename}")

            # Evaluate agent_e vs agent_f directly
            p0_avg, p1_avg = evaluate_agents(
                env, agent_e, agent_f, num_episodes=1000, plot=False
            )
            print("Over 1000 episodes of EveryVisit vs. FirstVisit (Orientation 1):")
            print(f"  EveryVisit (Player 0) average payoff = {p0_avg}")
            print(f"  FirstVisit (Player 1) average payoff = {p1_avg}")
            if p0_avg > p1_avg:
                print("  => EveryVisitMCAgent performs better.")
            elif p1_avg > p0_avg:
                print("  => FirstVisitMCAgent performs better.")
            else:
                print("  => They perform equally.")


# --------------------
# 2) NEW "flipped orientation": agent_f (FirstVisit) is Player 0, agent_e (EveryVisit) is Player 1
# --------------------
for sp in state_processors:
    for e in epsilons:
        for g in gammas:
            print(
                f"\n=== (Orientation 2) FirstVisit vs. EveryVisit "
                f"w/ e={e}, gamma={g}, sp={sp.__name__} ==="
            )

            # Create fresh agents
            agent_e = EveryVisitMCAgent(epsilon=e, gamma=g)
            agent_f = FirstVisitMCAgent(epsilon=e, gamma=g)

            # Train them with reversed orientation
            # agent_f is Player 0, agent_e is Player 1
            play_episodes(
                env,
                agent_f,  # Player 0
                agent_e,  # Player 1
                num_episodes=NUM_EPISODES,
                do_update=True,
                update_freq=UPDATE_FREQ,
                state_transformer=sp,
            )

            # Now evaluate agent_f (Player 0) vs Random
            f_vs_rand_payoffs = play_episodes(
                env, agent_f, random_agent, 10000, do_update=False
            )
            # As Player 0, payoff is at index [0]
            f_rewards = [p[0] for p in f_vs_rand_payoffs]
            f_cumulative = np.cumsum(f_rewards)

            # Evaluate agent_e (Player 1) vs Random
            e_vs_rand_payoffs = play_episodes(
                env, random_agent, agent_e, 10000, do_update=False
            )
            # As Player 1, payoff is at index [1]
            e_rewards = [p[1] for p in e_vs_rand_payoffs]
            e_cumulative = np.cumsum(e_rewards)

            # Plot both lines in a single figure
            plt.figure()
            plt.plot(f_cumulative, label="FirstVisit MC vs Random (Player 0)")
            plt.plot(e_cumulative, label="EveryVisit MC vs Random (Player 1)")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title(f"[Orientation 2] {sp.__name__}, e={e}, gamma={g}")
            plt.legend()

            # Save plot to file
            filename = f"graphs/orient2_FvE_{sp.__name__}_e{e}_g{g}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close()
            print(f"Saved combined plot to {filename}")

            # Evaluate agent_f vs agent_e directly, f is Player 0, e is Player 1
            p0_avg, p1_avg = evaluate_agents(
                env, agent_f, agent_e, num_episodes=1000, plot=False
            )
            print("Over 1000 episodes of FirstVisit vs. EveryVisit (Orientation 2):")
            print(f"  FirstVisit (Player 0) average payoff = {p0_avg}")
            print(f"  EveryVisit (Player 1) average payoff = {p1_avg}")
            if p0_avg > p1_avg:
                print("  => FirstVisitMCAgent performs better.")
            elif p1_avg > p0_avg:
                print("  => EveryVisitMCAgent performs better.")
            else:
                print("  => They perform equally.")
