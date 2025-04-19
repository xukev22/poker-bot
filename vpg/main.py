import rlcard
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .PolicyNetwork import VPGPolicy
from agents import PolicyAgent, RandomAgent, EveryVisitMCAgent
from utils import process_limit_state_v2, process_limit_state_v1

from experiments import evaluate_agents

# enable single‑agent mode so step() returns (next_state, reward)
env = rlcard.make("limit-holdem", config={"single_agent_mode": True})
state = env.reset()
state = state[0]
input_dim = len(state["obs"])
n_actions = env.num_actions

policy = VPGPolicy(input_dim, n_actions)
optimizer = Adam(policy.parameters(), lr=0.001)
gamma = 0.99
num_episodes = 10000

for ep in range(num_episodes):
    state = env.reset()
    state = state[0]
    log_probs = []
    rewards = []

    # roll out one episode
    while not env.is_over():
        obs = torch.tensor(state["obs"], dtype=torch.float32)
        logits = policy(obs)

        # mask illegal actions
        legal_ids = list(state["legal_actions"].keys())
        mask = torch.full((n_actions,), float("-1e9"))
        mask[legal_ids] = 0
        logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        # print(probs)
        action = torch.multinomial(probs, 1).item()
        log_probs.append(torch.log(probs[action]))

        # now step gives (next_state, reward)
        state, reward = env.step(action)
        rewards.append(reward)

    # compute reward‑to‑go and do one REINFORCE update
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # use ddof=0 so std of a single value (instant fold) is 0 (not nan)
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

    policy_loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # if (ep + 1) % 1000 == 0:
    #     print(f"Episode {ep+1:5d}, avg return: {returns.mean().item():.3f}")

# re‑create the holdem environment, prob not needed just being safe (multi‐agent mode)
eval_env = rlcard.make("limit-holdem")  # default is multi‑agent

# instantiate trained policy and a random agent
trained_agent = PolicyAgent(policy)  # VPGPolicy, already trained
random_agent = RandomAgent()

# raw acts
train, rand = evaluate_agents(
    eval_env,
    trained_agent,
    random_agent,
    10000,
    False,
    None,
)

print(train, rand)

agent_gen0_aggro_v2l = EveryVisitMCAgent.load("agents/agent_gen0_aggro_v2l.pkl")
agent_gen0_std_v2l = EveryVisitMCAgent.load("agents/agent_gen0_std_v2l.pkl")


g0av2l, vpg = evaluate_agents(
    env,
    agent_gen0_aggro_v2l,
    trained_agent,
    100000,
    False,
    process_limit_state_v2,
)
print("Eval payouts g0av2l vs vpg:", g0av2l, vpg)

vpg, g0sv2l = evaluate_agents(
    env,
    trained_agent,
    agent_gen0_std_v2l,
    100000,
    False,
    process_limit_state_v2,
)
print("Eval payouts vpg vs g0sv2l:", vpg, g0sv2l)


agent_gen0_aggro_v1l = EveryVisitMCAgent.load("agents/gen0_v1l_aggro.pkl")
agent_gen0_std_v1l = EveryVisitMCAgent.load("agents/gen0_v1l_std.pkl")

g0av1l, vpg = evaluate_agents(
    env,
    agent_gen0_aggro_v1l,
    trained_agent,
    100000,
    False,
    process_limit_state_v1,
)
print("Eval payouts g0av1l vs vpg:", g0av1l, vpg)

vpg, g0sv1l = evaluate_agents(
    env,
    trained_agent,
    agent_gen0_std_v1l,
    100000,
    False,
    process_limit_state_v1,
)
print("Eval payouts vpg vs g0sv1l:", vpg, g0sv1l)
