import rlcard
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .PolicyNetwork import VPGPolicy
from agents import PolicyAgent, HumanAgent

from experiments import human_play_bot

# enable single‑agent mode so step() returns (next_state, reward)
env = rlcard.make("limit-holdem", config={"single_agent_mode": True})
state = env.reset()
state = state[0]
input_dim = len(state["obs"])
n_actions = env.num_actions

policy = VPGPolicy(input_dim, n_actions)
optimizer = Adam(policy.parameters(), lr=0.001)
gamma = 0.99
num_episodes = 1000

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
human_agent = HumanAgent()

human_play_bot(eval_env, human_agent, trained_agent, False)
