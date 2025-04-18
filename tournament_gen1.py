import rlcard
from agents import EveryVisitMCAgent, RandomAgent
from experiments import play_episodes, evaluate_agents
from utils import process_leduc_state_v4

NUM_EPISODES = 1000000
UPDATE_FREQ = 100
EPSILON = 0.01

env = rlcard.make("leduc-holdem")
random_agent = RandomAgent()

# note:
# v4 -> betting history
# v1 -> stacks
# aggro acts first

# save =============================================================================================

agent_gen1_aggro_v4 = EveryVisitMCAgent.load("agents/agent_gen0_aggro_v4.pkl")
agent_gen1_std_v4 = EveryVisitMCAgent.load("agents/agent_gen0_std_v4.pkl")

print("Training agent_gen1_aggro_v4 vs. agent_gen1_std_v4 agent")
train_payoffs = play_episodes(
    env,
    agent_gen1_aggro_v4,
    agent_gen1_std_v4,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v4,
)

g1av4, r_g1av4 = evaluate_agents(
    env, agent_gen1_aggro_v4, random_agent, num_episodes=10000
)
r_g1sv4, g1sv4 = evaluate_agents(
    env, random_agent, agent_gen1_std_v4, num_episodes=10000
)

print("Eval payouts g1av4 vs random:", g1av4, r_g1av4)
print("Eval payouts g1sv4 vs random:", g1sv4, r_g1sv4)

print("Saving gen 1 v4 agents")
agent_gen1_aggro_v4.save("agents/agent_gen1_aggro_v4.pkl")
agent_gen1_std_v4.save("agents/agent_gen1_std_v4.pkl")

g1av4, g1sv4 = evaluate_agents(
    env, agent_gen1_aggro_v4, agent_gen1_std_v4, num_episodes=10000
)
print("Eval payouts g1av4 vs g1sv4:", g1av4, g1sv4)
