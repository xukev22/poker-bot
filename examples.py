import rlcard
from agents import FirstVisitMCAgent, RandomAgent, HumanAgent, EveryVisitMCAgent
from experiments import play_episodes, human_play_bot, evaluate_agents
from utils import (
    process_leduc_state_v1,
    process_leduc_state_v2,
    process_leduc_state_v3,
    process_leduc_state_v4,
)

# NUM_EPISODES = 1000000
# UPDATE_FREQ = 100

# agente0 = EveryVisitMCAgent(epsilon=0.1, gamma=0.1)
# agente1 = EveryVisitMCAgent(epsilon=0.01, gamma=1)

# env = rlcard.make("leduc-holdem")
# agent0 = FirstVisitMCAgent(
#     epsilon=0.01, gamma=0.5, state_transformer=process_leduc_state_v4
# )
# agent1 = FirstVisitMCAgent(
#     epsilon=0.01, gamma=0.5, state_transformer=process_leduc_state_v4
# )


# print(
#     f"Training the two MC agents vs each other for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v4 state"
# )
# train_payoffs = play_episodes(
#     env,
#     agent0,
#     agent1,
#     num_episodes=NUM_EPISODES,
#     do_update=True,
#     update_freq=UPDATE_FREQ,
#     state_transformer=process_leduc_state_v4,
# )

# env_eval = rlcard.make("leduc-holdem")
# agent_random = RandomAgent()

# avg_return, rand_return = evaluate_agents(
#     env_eval, agent0, agent_random, num_episodes=100000
# )
# print(
#     f"Trained agent0 vs Random -> avg payoff: {avg_return}, random payoff: {rand_return}"
# )

# avg_return, rand_return = evaluate_agents(
#     env_eval, agent_random, agent1, num_episodes=100000
# )
# print(
#     f"Trained agent1 vs Random -> avg payoff: {avg_return}, random payoff: {rand_return}"
# )

# avg_return0, avg_return1 = evaluate_agents(
#     env_eval, agent0, agent1, num_episodes=100000
# )
# print(
#     f"Trained agent0 vs agent1 -> avg0 payoff: {avg_return0}, random payoff: {avg_return1}"
# )


# env = rlcard.make("leduc-holdem")
# agent0 = EveryVisitMCAgent(
#     epsilon=0.1, gamma=0.5, state_transformer=process_leduc_state_v1
# )
# agent1 = FirstVisitMCAgent(
#     epsilon=0.1, gamma=0.5, state_transformer=process_leduc_state_v1
# )


# print(
#     f"Training the two MC agents vs each other for {NUM_EPISODES} episodes, updating every {UPDATE_FREQ} episodes... v1 state"
# )
# train_payoffs = play_episodes(
#     env,
#     agent0,
#     agent1,
#     num_episodes=NUM_EPISODES,
#     do_update=True,
#     update_freq=UPDATE_FREQ,
#     state_transformer=process_leduc_state_v1,
# )

# env_eval = rlcard.make("leduc-holdem")
# agent_random = RandomAgent()

# avg_return, rand_return = evaluate_agents(
#     env_eval, agent0, agent_random, num_episodes=10000
# )
# print(
#     f"Trained agent0 vs Random -> avg payoff: {avg_return}, random payoff: {rand_return}"
# )

# avg_return, rand_return = evaluate_agents(
#     env_eval, agent1, agent_random, num_episodes=10000
# )
# print(
#     f"Trained agent1 vs Random -> avg payoff: {avg_return}, random payoff: {rand_return}"
# )

# avg_return0, avg_return1 = evaluate_agents(env_eval, agent0, agent1, num_episodes=10000)
# print(
#     f"Trained agent0 vs agent1 -> avg0 payoff: {avg_return0}, random payoff: {avg_return1}"
# )


# print(agent0.Q)


# if you put the trained bot on wrong permutation it may get confused
env_human = rlcard.make("leduc-holdem")
human_agent = HumanAgent()
new_agent0 = FirstVisitMCAgent.load("agents/agent0_2.pkl")
new_agent1 = FirstVisitMCAgent.load("agents/agent1_2.pkl")
# human_play_bot(env_human, human_agent, new_agent1, True)
human_play_bot(env_human, human_agent, new_agent0, False)
