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
UPDATE_FREQ = 10

env = rlcard.make("leduc-holdem")
random_agent = RandomAgent()

agent0 = FirstVisitMCAgent(
    epsilon=0.01, gamma=0.9, state_transformer=process_leduc_state_v3
)
agent1 = FirstVisitMCAgent(
    epsilon=0.01, gamma=0.5, state_transformer=process_leduc_state_v3
)

print(
    f"Training the FVMC0 agent vs random for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v3 state"
)
train_payoffs = play_episodes(
    env,
    agent0,
    random_agent,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v3,
)


print(
    f"Training the FVMC1 agent vs random for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v3 state"
)
train_payoffs = play_episodes(
    env,
    random_agent,
    agent1,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v3,
)

agent0.save("agents/agent0_1.pkl")
agent1.save("agents/agent1_1.pkl")

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
    f"Training the TRAINED FVMC0 agent vs TRAINED FVMC1 for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v3 state"
)
train_payoffs = play_episodes(
    env,
    agent0,
    agent1,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v3,
)

agent0.save("agents/agent0_2.pkl")
agent1.save("agents/agent1_2.pkl")

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

old_agent0 = FirstVisitMCAgent.load("agents/agent0_1.pkl")
old_agent1 = FirstVisitMCAgent.load("agents/agent1_1.pkl")

new_agent0 = FirstVisitMCAgent.load("agents/agent0_2.pkl")
new_agent1 = FirstVisitMCAgent.load("agents/agent1_2.pkl")

env_eval = rlcard.make("leduc-holdem")
avg_return0, avg_return1 = evaluate_agents(
    env_eval, old_agent0, new_agent1, num_episodes=10000
)
print(
    f"Trained old agent0 vs new agent1 -> avg0 payoff: {avg_return0}, avg1 payoff: {avg_return1}"
)

env_eval = rlcard.make("leduc-holdem")
avg_return0, avg_return1 = evaluate_agents(
    env_eval, new_agent0, old_agent1, num_episodes=10000
)
print(
    f"Trained new agent0 vs old agent1 -> avg0 payoff: {avg_return0}, avg1 payoff: {avg_return1}"
)

env_eval = rlcard.make("leduc-holdem")
avg_return, rand_return = evaluate_agents(
    env_eval, new_agent0, random_agent, num_episodes=10000
)
print(
    f"Trained new agent0 vs Random -> avg payoff: {avg_return}, random payoff: {rand_return}"
)
env_eval = rlcard.make("leduc-holdem")
avg_return, rand_return = evaluate_agents(
    env_eval, random_agent, new_agent1, num_episodes=10000
)
print(
    f"Trained new agent1 vs Random -> random payoff: {avg_return}, avg payoff: {rand_return}"
)

agent3 = FirstVisitMCAgent(
    epsilon=0.01, gamma=1, state_transformer=process_leduc_state_v3
)

print(
    f"Training the FVMC3 agent vs TRAINED FVMC0 agent for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v3 state"
)
train_payoffs = play_episodes(
    env,
    new_agent0,
    agent3,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v3,
)
env_eval = rlcard.make("leduc-holdem")
avg0_return, avg3_return = evaluate_agents(
    env_eval, new_agent0, agent3, num_episodes=10000
)
print(
    f"Trained agent3 vs new agent 0 -> avg0 payoff: {avg0_return}, avg3 payoff: {avg3_return}"
)

env_eval = rlcard.make("leduc-holdem")
avg0_return, rand_return = evaluate_agents(
    env_eval, random_agent, agent3, num_episodes=10000
)
print(
    f"Trained new agent3 vs Random -> random payoff: {avg0_return}, avg payoff: {rand_return}"
)
