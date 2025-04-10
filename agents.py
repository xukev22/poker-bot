import random
from utils import process_leduc_state_v1
from collections import defaultdict


class RandomAgent:
    """
    Picks random actions in the env.
    """

    def step(self, state):
        raw_obs_dict = state["raw_obs"]
        legal_acts = raw_obs_dict["legal_actions"]

        return random.choice(legal_acts)

    def update(self, trajectories):
        pass


class FirstVisitMCAgent:
    """
    A First‑Visit Monte Carlo control agent for imperfect‑information games (e.g. Leduc Poker).

    Attributes:
        epsilon (float): Exploration probability for e‑greedy action selection.
        gamma (float): Discount factor for future rewards.
        Q (defaultdict): Action‑value estimates, Q[state][action] -> float.
        N (defaultdict): Counts of first visits, N[state][action] -> int.
    """

    def __init__(self, epsilon=0.1, gamma=0.9):
        """
        Initialize the agent.

        Args:
            epsilon (float): Probability of choosing a random action (exploration).
            gamma (float): Discount factor for computing returns.
        """
        self.epsilon = epsilon
        self.gamma = gamma

        self.Q = defaultdict(lambda: defaultdict(float))
        self.N = defaultdict(lambda: defaultdict(int))

    def step(self, state):
        """
        Choose an action in the given state using e‑greedy over legal actions.

        Exploration: with probability epsilon, pick uniformly from legal actions.
        Exploitation: otherwise, pick the legal action(s) with highest Q‑value
        (ties broken uniformly at random).

        Args:
            state (dict): The environment’s raw state for the current player. Must contain:
                - state["raw_obs"]["legal_actions"]: list of legal action labels.
                - state["raw_obs"]["current_player"]: integer player index.

        Returns:
            action: One of the legal actions.
        """
        raw_obs_dict = state["raw_obs"]
        legal_acts = raw_obs_dict["legal_actions"]
        cur_pid = raw_obs_dict["current_player"]

        # uses e-greedy
        if random.random() < self.epsilon:
            # explore
            return random.choice(legal_acts)

        # exploit break ties randomly
        info_s = process_leduc_state_v1(state, cur_pid)
        q_s = self.Q[info_s]  # defaultdict(float), so unseen (s,a) is 0.0

        # find max Q among legal actions
        best_value = None
        best_acts = []
        for a in legal_acts:
            v = q_s[a]
            if (best_value is None) or (v > best_value):
                best_value = v
                best_acts = [a]
            elif v == best_value:
                best_acts.append(a)

        return random.choice(best_acts)

    def update(self, trajectories):
        """
        Perform First‑Visit Monte Carlo updates on completed episode(s).

        Each trajectory is a list of (state, action, reward) tuples for one player,
        where reward is typically zero for all but the final step.

        For each trajectory:
          - Traverse backwards to compute the return G at each time-step.
          - For each (state, action), if it’s the first occurrence in that episode,
            increment N[state][action], and update Q[state][action] via:
                Q += (1 / N) * (G - Q)

        Args:
            trajectories (List[List[Tuple[state, action, reward]]]):
                A list of one or more episodes for a single player. Each episode is
                in chronological order:
                    [
                        (s0, a0, r1),
                        (s1, a1, r2),
                        …,
                        (s_{T-1}, a_{T-1}, r_T)
                    ]
        """
        for episode in trajectories:
            G = 0.0
            first_visit = set()
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = self.gamma * G + r
                if (s, a) not in first_visit:
                    first_visit.add((s, a))
                    self.N[s][a] += 1
                    alpha = 1.0 / self.N[s][a]
                    self.Q[s][a] += alpha * (G - self.Q[s][a])
