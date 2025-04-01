import numpy as np
from base_agent import BaseAgent


class FirstVisitMC(BaseAgent):
    def __init__(self, action_num, epsilon=0.1, gamma=0.9):
        """
        Initializes the First-Visit Monte Carlo agent.

        :param action_num: Total number of available actions.
        :param epsilon: The probability of selecting a random action (exploration).
        :param gamma: Discount factor for future rewards.
        """
        self.action_num = action_num
        self.epsilon = epsilon
        self.gamma = gamma

        self.q = dict()  # (s,a) -> Q value
        self.n = dict()  # (s, a) -> counts
        self.episode_memory = []

    def _get_key(self, state, action):
        """
        Helper method to create a hashable key from state and action.
        Here, we simply use the string representation of the state.
        """
        return (repr(state), action)

    def step(self, state):
        """
        Selects an action during training using an e-greedy strategy.
        This method is called by the environment when in training mode.

        :param state: A dictionary containing the current state and legal actions.
        :return: Selected action.
        """
        legal_actions = state["legal_actions"]

        # roll for e
        if np.random.rand() < self.epsilon:
            # explore
            return np.random.choice(legal_actions)
        else:
            best_action = None
            best_value = -float("inf")
            for action in legal_actions:
                key = self._get_key(state, action)
                value = self.Q.get(key, 0.0)  # Default Q-value is 0.0 if unseen
                if value > best_value:
                    best_value = value
                    best_action = action
            # if none of the actions have been evaluated yet, pick randomly
            return (
                best_action
                if best_action is not None
                else np.random.choice(legal_actions)
            )

    def eval_step(self, state):
        """
        Selects an action during evaluation in a deterministic (greedy) way.
        This method is used when the environment is in evaluation mode.

        :param state: A dictionary containing the current state and legal actions.
        :return: Selected action.
        """
        legal_actions = state["legal_actions"]
        best_action = None
        best_value = -float("inf")
        for action in legal_actions:
            key = self._get_key(state, action)
            value = self.Q.get(key, 0.0)
            if value > best_value:
                best_value = value
                best_action = action
        # note we pick the first legal action for deterministic return unlike in non eval step
        return best_action if best_action is not None else legal_actions[0]

    def feed(self, ts):
        """
        Processes a transition and updates the agent.
        Transitions are stored until the end of the episode (i.e., when done==True),
        at which point we perform a first-visit MC update for each state-action pair.

        :param ts: A tuple (state, action, reward, next_state, done).
        """
        self.episode_memory.append(ts)
        state, action, reward, next_state, done = ts

        # if the episode is finished, perform the first-visit MC update.
        if done:
            seen = set()
            # iterate over the episode transitions in order
            for i, (s, a, r, s_next, done_flag) in enumerate(self.episode_memory):
                key = self._get_key(s, a)
                # process only the first occurrence of this state-action pair
                if key not in seen:
                    seen.add(key)
                    # compute the return G from time step i to the end of the episode
                    G = sum(
                        self.gamma ** (j - i) * self.episode_memory[j][2]
                        for j in range(i, len(self.episode_memory))
                    )
                    # incrementally update Q(s, a)
                    self.N[key] = self.N.get(key, 0) + 1
                    self.Q[key] = (
                        self.Q.get(key, 0.0) + (G - self.Q.get(key, 0.0)) / self.N[key]
                    )
            # clear the episode memory for the next episode
            self.episode_memory = []
