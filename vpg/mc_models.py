import rlcard
from agents import EveryVisitMCAgent, RandomAgent
from experiments import play_episodes, evaluate_agents
from utils import process_limit_state_v2

NUM_EPISODES = 1000000
UPDATE_FREQ = 100
EPSILON = 0.01

env = rlcard.make("limit-holdem")
random_agent = RandomAgent()

# save =============================================================================================

agent_gen0_aggro_v2l = EveryVisitMCAgent(
    epsilon=EPSILON, gamma=0.1, state_transformer=process_limit_state_v2
)
agent_gen0_std_v2l = EveryVisitMCAgent(
    epsilon=EPSILON, gamma=0.99, state_transformer=process_limit_state_v2
)

print("Training agent_gen0_aggro_v2l vs. random agent")
train_payoffs = play_episodes(
    env,
    agent_gen0_aggro_v2l,
    random_agent,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_limit_state_v2,
)

print("Training agent_gen0_std_v2l vs. random agent")
train_payoffs = play_episodes(
    env,
    random_agent,
    agent_gen0_std_v2l,
    num_episodes=NUM_EPISODES,
    do_update=True,
    update_freq=UPDATE_FREQ,
    state_transformer=process_limit_state_v2,
)

g0av2l, r_g0av2l = evaluate_agents(
    env, agent_gen0_aggro_v2l, random_agent, num_episodes=10000
)
r_g0sv2l, g0sv2l = evaluate_agents(
    env, random_agent, agent_gen0_std_v2l, num_episodes=10000
)

print("Eval payouts g0av2l vs random:", g0av2l, r_g0av2l)
print("Eval payouts g0sv2l vs random:", g0sv2l, r_g0sv2l)

print("Saving gen 0 v2l agents")
agent_gen0_aggro_v2l.save("agents/agent_gen0_aggro_v2l.pkl")
agent_gen0_std_v2l.save("agents/agent_gen0_std_v2l.pkl")
