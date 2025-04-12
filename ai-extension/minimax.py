from leduc_game import LeducHoldem
from enum import Enum
import copy


# state = env.get_perfect_information()
# print(state)
# print(state["legal_actions"])
# print(env.game.dealer.deck)
# for card in env.game.dealer.deck:
#     print(str(card))

# # Assuming env.game.players is a list of Player objects
# for i, player in enumerate(env.game.players):
#     print(player.hand)

env = LeducHoldem()


class NodeType(Enum):
    MAX = 1
    MIN = 2
    CHANCE = 3


def expectimax(env, depth, nodeType):
    # extract state from env
    state = env.get_perfect_information()

    # base case: if terminal state or depth reached
    if env.is_over() or depth == 0:
        return evaluate(state)

    # max case
    if nodeType == NodeType.MAX:
        value = float("-inf")
        for action in state["legal_actions"]:
            # copy environment and make move on copied env
            env_copy = copy.deepcopy(env)
            env_copy.step(action, True)

            # extract value
            value = max(value, expectimax(env_copy, depth - 1, NodeType.MIN))
        return value
    # min case
    elif nodeType == NodeType.MIN:
        value = float("inf")
        for action in state["legal_actions"]:
            # copy environment and make move on copied env
            env_copy = copy.deepcopy(env)
            env_copy.step(action, True)

            # extract value
            value = min(value, expectimax(env_copy, depth - 1, NodeType.MAX))
        return value
    # chance case
    elif nodeType == NodeType.CHANCE:
        # for each possible chance outcome, we compute the expected value
        # based on the probability of each outcome
        value = 0
        for outcome_pair in possible_outcomes(state):
            outcome = outcome_pair[0]
            probability = outcome_pair[1]
            # newState = applyOutcome(state, outcome)
            # value += p * expectiminimax(newState, depth - 1, NEXT_NODE)
        return value


def evaluate(state):
    pass


def possible_outcomes(state):
    pass
