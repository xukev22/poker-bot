import rlcard
from agents import EveryVisitMCAgent, RandomAgent
from experiments import play_episodes, evaluate_agents
from utils import process_limit_state_v1, process_limit_state_v2

env = rlcard.make("limit-holdem")
random_agent = RandomAgent()

env.reset()

# TRAIN
state_for_pid = env.get_state(env.get_player_id())


agent_ev4 = EveryVisitMCAgent(
    epsilon=0.01, gamma=1.0, state_transformer=process_limit_state_v2
)

print("Training agent_ev4 vs. random agent")
train_payoffs = play_episodes(
    env,
    agent_ev4,
    random_agent,
    num_episodes=10000000,
    do_update=True,
    update_freq=100,
    state_transformer=process_limit_state_v2,
)
agent_ev4.save("agents/agentR1e7v2.pkl")
agentR, randomR = evaluate_agents(env, agent_ev4, random_agent, num_episodes=100000)
print(agentR, randomR)


# LOAD
# agent_ev4 = EveryVisitMCAgent.load("agents/agentR.pkl")
# agentR, randomR = evaluate_agents(env, agent_ev4, random_agent, num_episodes=100000)
# print(agentR, randomR)

