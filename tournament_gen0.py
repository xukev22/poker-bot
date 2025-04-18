import rlcard
from agents import EveryVisitMCAgent, RandomAgent, HumanAgent
from experiments import play_episodes, human_play_bot, evaluate_agents
from utils import process_leduc_state_v4

NUM_EPISODES = 1000000
UPDATE_FREQ = 100
EPSILON = 0.01

env = rlcard.make("leduc-holdem")
random_agent = RandomAgent()

# note:
# v4 -> betting history
# v1 -> stacks

# save =============================================================================================

# agent_gen0_aggro_v4 = EveryVisitMCAgent(
#     epsilon=EPSILON, gamma=0.1, state_transformer=process_leduc_state_v4
# )
# agent_gen0_std_v4 = EveryVisitMCAgent(
#     epsilon=EPSILON, gamma=0.99, state_transformer=process_leduc_state_v4
# )

# print("Training agent_gen0_aggro_v4 vs. random agent")
# train_payoffs = play_episodes(
#     env,
#     agent_gen0_aggro_v4,
#     random_agent,
#     num_episodes=NUM_EPISODES,
#     do_update=True,
#     update_freq=UPDATE_FREQ,
#     state_transformer=process_leduc_state_v4,
# )

# print("Training agent_gen0_std_v4 vs. random agent")
# train_payoffs = play_episodes(
#     env,
#     random_agent,
#     agent_gen0_std_v4,
#     num_episodes=NUM_EPISODES,
#     do_update=True,
#     update_freq=UPDATE_FREQ,
#     state_transformer=process_leduc_state_v4,
# )

# g0av4, r_g0av4 = evaluate_agents(
#     env, agent_gen0_aggro_v4, random_agent, num_episodes=10000
# )
# r_g0sv4, g0sv4 = evaluate_agents(
#     env, random_agent, agent_gen0_std_v4, num_episodes=10000
# )

# print("Eval payouts g0av4 vs random:", g0av4, r_g0av4)
# print("Eval payouts g0sv4 vs random:", g0sv4, r_g0sv4)

# print("Saving gen 0 v4 agents")
# agent_gen0_aggro_v4.save("agents/agent_gen0_aggro_v4.pkl")
# agent_gen0_std_v4.save("agents/agent_gen0_std_v4.pkl")

# load ==========================================================================================
agent_gen0_aggro_v4 = EveryVisitMCAgent.load("agents/agent_gen0_aggro_v4.pkl")
agent_gen0_std_v4 = EveryVisitMCAgent.load("agents/agent_gen0_std_v4.pkl")

g0av4, r_g0av4 = evaluate_agents(
    env, agent_gen0_aggro_v4, random_agent, num_episodes=10000
)
r_g0sv4, g0sv4 = evaluate_agents(
    env, random_agent, agent_gen0_std_v4, num_episodes=10000
)

print("Eval payouts g0av4 vs random:", g0av4, r_g0av4)
print("Eval payouts g0sv4 vs random:", g0sv4, r_g0sv4)

g0av4, g0sv4 = evaluate_agents(
    env, agent_gen0_aggro_v4, agent_gen0_std_v4, num_episodes=10000
)
print("Eval payouts g0av4 vs g0sv4:", g0av4, g0sv4)

g0sv4, g0av4 = evaluate_agents(
    env, agent_gen0_std_v4, agent_gen0_aggro_v4, num_episodes=10000
)
print("(switched) Eval payouts g0av4 vs g0sv4:", g0av4, g0sv4)
