import rlcard
from enum import Enum
import copy

# env = rlcard.make("leduc-holdem")
# env.reset()

# state = env.get_perfect_information()
# print(state)
# print(state["legal_actions"])
# print(env.game.dealer.deck)
# for card in env.game.dealer.deck:
#     print(str(card))

# # Assuming env.game.players is a list of Player objects
# for i, player in enumerate(env.game.players):
#     print(player.hand)

# init_pid = env.get_player_id()
# print(init_pid)


# Helper function to decide which branch node to use at the next state.
# It uses the current state's "current_player" to check if it's the same as sb_id.
def get_branch_node_type(state, sb_id):
    if state["current_player"] == sb_id:
        return NodeType.MAX
    else:
        return NodeType.MIN


# Detect if the current state is a chance node.
# For instance, in Leduc Hold’em a chance node may occur in the round where the community card is dealt.
# Here we assume that if current_round is 1 and the public card hasn't been dealt (None),
# then we are in a chance node.
def is_chance_node(env):
    info = env.get_perfect_information()
    return info["current_round"] == 1 and info["public_card"] is None


# Simple evaluation heuristic.
# From the perspective of sb_id, it returns the difference in chip count.
# A positive value means a favorable outcome for the agent.
def evaluate(state, sb_id):
    chips = state.get("chips", [0, 0])
    return chips[sb_id] - chips[1 - sb_id]


# Advanced evaluation heuristic.
# Combines chip difference with an estimation of hand strength.
def advanced_evaluate(state, sb_id):
    # First, evaluate chip counts.
    chips = state.get("chips", [0, 0])
    chip_diff = chips[sb_id] - chips[1 - sb_id]

    # Initialize hand strength.
    hand_strength = 0

    # Retrieve hand cards and public card.
    hand_cards = state.get("hand_cards", [])
    public_card = state.get("public_card")

    # Define a rank ordering.
    # For Leduc Hold’em we typically use J, Q, and K.
    rank_value = {"J": 1, "Q": 2, "K": 3}

    if public_card is not None:
        # Postflop evaluation: public card is on the board.

        # Check if there is a pair between a hole card and the public card.
        for card in hand_cards:
            if card[-1] == public_card[-1]:
                hand_strength += 5  # Bonus for matching rank with the board.

        # If both hole cards form a pair, add a stronger bonus.
        if len(hand_cards) >= 2 and hand_cards[0][-1] == hand_cards[1][-1]:
            hand_strength += 10

        # Also add the highest rank among the hand and public card as a tiebreaker.
        max_rank = 0
        for card in hand_cards:
            r = rank_value.get(card[-1], 0)
            max_rank = max(max_rank, r)
        max_rank = max(max_rank, rank_value.get(public_card[-1], 0))
        hand_strength += max_rank
    else:
        # Preflop evaluation: no public card yet.
        # Use only the hole cards to gauge strength.
        max_rank = 0
        for card in hand_cards:
            r = rank_value.get(card[-1], 0)
            max_rank = max(max_rank, r)
        hand_strength += max_rank

    # Combine the chip differential and hand strength.
    # The weighting factors can be tuned. Here, both factors contribute equally.
    evaluation = 0.5 * chip_diff + 0.5 * hand_strength
    return evaluation


# Generates possible outcomes for a chance event.
# Here, each card in the current deck becomes a possible outcome.
def possible_outcomes(env):
    outcomes = []
    # Work on a copy of the deck to avoid unintended side effects.
    deck = env.game.dealer.deck[:]
    total = len(deck)
    for card in deck:
        new_env = copy.deepcopy(env)
        # Remove the card being dealt. (If your game logic requires additional state updates—such as
        # adding the card to a board—add that here.)
        new_env.game.dealer.deck.remove(card)
        outcomes.append((new_env, 1 / total))
    return outcomes


class NodeType(Enum):
    MAX = 1
    MIN = 2
    CHANCE = 3


# The expectimax function now takes an explicit nodeType and uses sb_id to switch between decision branches.
def expectimax(env, depth, nodeType, sb_id):
    state = env.get_perfect_information()

    # Base case: terminal game state or maximum search depth reached.
    if env.is_over() or depth == 0:
        # print(advanced_evaluate(state, sb_id))
        return advanced_evaluate(state, sb_id)

    # If the current node is a chance node (or the state indicates a chance event), handle it.
    if nodeType == NodeType.CHANCE or is_chance_node(env):
        value = 0
        outcomes = possible_outcomes(env)
        for new_env, probability in outcomes:
            # After a chance event, deduce whether the new state's decision node is MAX or MIN.
            next_state = new_env.get_perfect_information()
            next_node_type = get_branch_node_type(next_state, sb_id)
            value += probability * expectimax(new_env, depth - 1, next_node_type, sb_id)
        return value

    # If this is a decision node (MAX or MIN), iterate through legal actions.
    elif nodeType == NodeType.MAX:
        value = float("-inf")
        for action in state["legal_actions"]:
            # Copy the environment to simulate the action.
            env_copy = copy.deepcopy(env)
            env_copy.step(action, True)
            next_state = env_copy.get_perfect_information()
            # Determine the next node type using the new state.
            next_node_type = get_branch_node_type(next_state, sb_id)
            value = max(value, expectimax(env_copy, depth - 1, next_node_type, sb_id))
        return value

    elif nodeType == NodeType.MIN:
        value = float("inf")
        for action in state["legal_actions"]:
            env_copy = copy.deepcopy(env)
            env_copy.step(action, True)
            next_state = env_copy.get_perfect_information()
            next_node_type = get_branch_node_type(next_state, sb_id)
            value = min(value, expectimax(env_copy, depth - 1, next_node_type, sb_id))
        return value


def get_best_action(env, depth, sb_id):
    best_action = None
    best_value = float("-inf")
    state = env.get_perfect_information()

    for action in state["legal_actions"]:
        env_copy = copy.deepcopy(env)
        env_copy.step(action, True)
        next_state = env_copy.get_perfect_information()
        next_node_type = get_branch_node_type(next_state, sb_id)

        # Evaluate the expectimax value after taking this action.
        value = expectimax(env_copy, depth - 1, next_node_type, sb_id)

        if value > best_value:
            best_value = value
            best_action = action

    return best_action, best_value


env = rlcard.make("leduc-holdem")
env.reset()

state = env.get_perfect_information()
print(state)

init_pid = env.get_player_id()

search_depth = 10  # or any search depth we prefer

best_action, best_value = get_best_action(env, search_depth, init_pid)
print("Best action:", best_action, "with expectimax value:", best_value)
