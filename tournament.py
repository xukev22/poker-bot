import rlcard
from agents import FirstVisitMCAgent, RandomAgent, HumanAgent, EveryVisitMCAgent
from experiments import play_episodes, human_play_bot, evaluate_agents
from utils import (
    process_leduc_state_v1,
    process_leduc_state_v2,
    process_leduc_state_v3,
    process_leduc_state_v4,
)

NUM_EPISODES = 1000000
UPDATE_FREQ = 25

env = rlcard.make("leduc-holdem")
random_agent = RandomAgent()

agent0 = FirstVisitMCAgent(
    epsilon=0.01, gamma=0.9, state_transformer=process_leduc_state_v1
)
agent1 = FirstVisitMCAgent(
    epsilon=0.01, gamma=0.5, state_transformer=process_leduc_state_v1
)

print(
    f"Training the FVMC0 agent vs random for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v4 state"
)
train_payoffs = play_episodes(
    env,
    agent0,
    random_agent,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v1,
)

print(
    f"Training the FVMC1 agent vs random for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v4 state"
)
train_payoffs = play_episodes(
    env,
    random_agent,
    agent1,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v1,
)

env_eval = rlcard.make("leduc-holdem")
avg_return, rand_return = evaluate_agents(
    env_eval, agent0, random_agent, num_episodes=10000
)
print(
    f"Trained agent0 vs Random -> avg payoff: {avg_return}, random payoff: {rand_return}"
)
env_eval = rlcard.make("leduc-holdem")
avg_return, rand_return = evaluate_agents(
    env_eval, random_agent, agent1, num_episodes=10000
)
print(
    f"Trained agent1 vs Random -> random payoff: {avg_return}, avg payoff: {rand_return}"
)

print(
    f"Training the TRAINED FVMC0 agent vs TRAINED FVMC1 for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v4 state"
)
train_payoffs = play_episodes(
    env,
    agent0,
    agent1,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v1,
)

env_eval = rlcard.make("leduc-holdem")
avg_return0, avg_return1 = evaluate_agents(env_eval, agent0, agent1, num_episodes=10000)
print(
    f"Trained agent0 vs agent1 -> avg0 payoff: {avg_return0}, avg1 payoff: {avg_return1}"
)

env_eval = rlcard.make("leduc-holdem")
avg_return, rand_return = evaluate_agents(
    env_eval, agent0, random_agent, num_episodes=10000
)
print(
    f"Trained agent0 vs Random -> avg payoff: {avg_return}, random payoff: {rand_return}"
)
env_eval = rlcard.make("leduc-holdem")
avg_return, rand_return = evaluate_agents(
    env_eval, random_agent, agent1, num_episodes=10000
)
print(
    f"Trained agent1 vs Random -> random payoff: {avg_return}, avg payoff: {rand_return}"
)
