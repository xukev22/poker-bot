from utils import process_leduc_state_v4
import rlcard
from agents import FirstVisitMCAgent, RandomAgent
from experiments import play_episodes, evaluate_agents_with_action_counts

# config
NUM_EPISODES = 1000000
UPDATE_FREQ = 100
SP = process_leduc_state_v4
EPSILON = 0.01

# testing over gammas
gammas = [1, 0.99, 0.9, 0.8, 0.5, 0.25, 0.1, 0.01]

env = rlcard.make("leduc-holdem")
random_agent = RandomAgent()

for g in gammas:
    print("gamma:", g)

    agent = FirstVisitMCAgent(epsilon=EPSILON, gamma=g, state_transformer=SP)

    # Train them against each other
    play_episodes(
        env,
        agent,
        random_agent,
        num_episodes=NUM_EPISODES,
        do_update=True,
        update_freq=UPDATE_FREQ,
        state_transformer=SP,
    )

    (avg0, avg1), counts = evaluate_agents_with_action_counts(
        env,
        agent,
        random_agent,
        num_episodes=10000,
        plot=True,
    )

    print(f"Average payoffs: Player0={avg0:.3f}, Player1={avg1:.3f}")
    print("Action counts for Player 0:", counts[0])
    print("Action counts for Player 1:", counts[1])
