import random
from utils import process_leduc_state_v1
from collections import defaultdict
import pickle

# pickle dont accept lambda

def default_float_dict():
    return defaultdict(float)


def default_int_dict():
    return defaultdict(int)


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


class HumanAgent:
    """UI to allow humans to interact with bots"""

    def step(self, state):
        raw_obs_dict = state["raw_obs"]
        legal_acts = raw_obs_dict["legal_actions"]

        while True:
            print(raw_obs_dict)
            print("Legal actions are:", ", ".join(legal_acts))
            user_input = input("Please enter one of the legal actions: ").strip()
            if user_input in legal_acts:
                action = user_input
                break
            else:
                print(f"❌  '{user_input}' is not valid. Try again.\n")

        # now `action` is guaranteed to be one of legal_acts
        return action

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

    def __init__(
        self, epsilon=0.1, gamma=0.9, state_transformer=process_leduc_state_v1
    ):
        """
        Initialize the agent.

        Args:
            epsilon (float): Probability of choosing a random action (exploration).
            gamma (float): Discount factor for computing returns.
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_transformer = state_transformer

        self.Q = defaultdict(default_float_dict)
        self.N = defaultdict(default_int_dict)

    def save(self, filepath):
        """Save the current agent to the given file path."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load an agent from the given file path."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def step(self, state, greedy=False):
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

        # Epsilon-greedy exploration
        if (not greedy) and (random.random() < self.epsilon):
            # explore
            return random.choice(legal_acts)

        # exploit break ties randomly
        info_s = self.state_transformer(state, cur_pid)
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


class EveryVisitMCAgent:
    """
    An Every‑Visit Monte Carlo control agent for imperfect‑information games (e.g. Leduc Poker).

    In contrast to First‑Visit MC, this agent updates its Q‑value estimates for every occurrence of
    a (state, action) pair in an episode.

    Attributes:
        epsilon (float): Exploration probability for epsilon‑greedy action selection.
        gamma (float): Discount factor for computing returns.
        Q (defaultdict): Action‑value estimates, Q[state][action] -> float.
        N (defaultdict): Counts of visits, N[state][action] -> int.
    """

    def __init__(
        self, epsilon=0.1, gamma=0.9, state_transformer=process_leduc_state_v1
    ):
        """
        Initialize the every-visit MC agent.

        Args:
            epsilon (float): Probability of choosing a random action (exploration).
            gamma (float): Discount factor for computing returns.
            state_transformer (callable): Function to transform raw states into a usable format.
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_transformer = state_transformer
        self.Q = defaultdict(default_float_dict)
        self.N = defaultdict(default_int_dict)

    def save(self, filepath):
        """Save the current agent to the given file path."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load an agent from the given file path."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def step(self, state, greedy=False):
        """
        Choose an action in the given state using an epsilon‑greedy policy.

        Args:
            state (dict): The current game state.
                Must contain:
                  - state["raw_obs"]["legal_actions"]: list of legal action labels.
                  - state["raw_obs"]["current_player"]: integer player index.
            greedy (bool): If True, select the best (greedy) action; otherwise, use epsilon‑greedy exploration.

        Returns:
            action: Chosen legal action.
        """
        raw_obs = state["raw_obs"]
        legal_acts = raw_obs["legal_actions"]
        cur_pid = raw_obs["current_player"]

        # Epsilon-greedy exploration
        if (not greedy) and (random.random() < self.epsilon):
            return random.choice(legal_acts)

        # exploitation: select the action with the highest Q-value (ties broken randomly)
        info_s = self.state_transformer(state, cur_pid)
        q_s = self.Q[info_s]

        # find max Q among legal actions
        best_value = None
        best_actions = []
        for a in legal_acts:
            v = q_s[a]
            if best_value is None or v > best_value:
                best_value = v
                best_actions = [a]
            elif v == best_value:
                best_actions.append(a)

        return random.choice(best_actions)

    def update(self, trajectories):
        """
        Update Q-values using the Every‑Visit MC rule.

        For each episode in trajectories:
          - Compute the return G at each timestep by traversing the episode in reverse.
          - Update Q[s][a] for every occurrence of (state, action).

        Args:
            trajectories (List[List[Tuple[state, action, reward]]]):
                A list of episodes, where each episode is a list of (state, action, reward) tuples.
        """
        for episode in trajectories:
            G = 0.0
            # Process the episode backwards (from terminal state)
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = self.gamma * G + r
                # For Every‑Visit MC, update every occurrence of (s, a)
                self.N[s][a] += 1
                alpha = 1.0 / self.N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])
