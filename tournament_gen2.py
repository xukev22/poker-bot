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

agent_gen2_aggro_v4 = EveryVisitMCAgent.load("agents/agent_gen1_aggro_v4.pkl")
agent_gen2_std_v4 = EveryVisitMCAgent.load("agents/agent_gen1_std_v4.pkl")

print("Training agent_gen2_aggro_v4 vs. agent_gen2_std_v4 agent")
train_payoffs = play_episodes(
    env,
    agent_gen2_aggro_v4,
    agent_gen2_std_v4,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_leduc_state_v4,
)

g2av4, r_g2av4 = evaluate_agents(
    env, agent_gen2_aggro_v4, random_agent, num_episodes=10000
)
r_g2sv4, g2sv4 = evaluate_agents(
    env, random_agent, agent_gen2_std_v4, num_episodes=10000
)

print("Eval payouts g2av4 vs random:", g2av4, r_g2av4)
print("Eval payouts g2sv4 vs random:", g2sv4, r_g2sv4)

print("Saving gen 2 v4 agents")
agent_gen2_aggro_v4.save("agents/agent_gen2_aggro_v4.pkl")
agent_gen2_std_v4.save("agents/agent_gen2_std_v4.pkl")

g2av4, g2sv4 = evaluate_agents(
    env, agent_gen2_aggro_v4, agent_gen2_std_v4, num_episodes=10000
)
print("Eval payouts g2av4 vs g2sv4:", g2av4, g2sv4)
