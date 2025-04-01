# defines an abstract base class that enforces the structure expected by RLCard
# any custom agent should subclass BaseAgent and implement the abstract methods

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, action_num):
        """
        Initializes the agent.
        :param action_num: The total number of actions available in the environment.
        """
        self.action_num = action_num

    @abstractmethod
    def step(self, state):
        """
        Given a state, select an action to take.
        This function is used during training and may include exploration.
        
        :param state: A dictionary representing the current state. Typically includes 'legal_actions'.
        :return: The selected action.
        """
        pass

    @abstractmethod
    def eval_step(self, state):
        """
        Given a state, select an action to take during evaluation.
        This should typically be a deterministic choice.
        
        :param state: A dictionary representing the current state.
        :return: The selected action.
        """
        pass

    @abstractmethod
    def feed(self, ts):
        """
        Process a transition tuple from the environment to update the agent.
        Typically, ts is a tuple of (state, action, reward, next_state, done).
        
        :param ts: A tuple containing (state, action, reward, next_state, done).
        """
        pass
