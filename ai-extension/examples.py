import rlcard
from enum import Enum
import copy

env = rlcard.make("leduc-holdem")
# env.reset()

# state = env.get_perfect_information()
# init_pid = env.get_player_id()

# print(state)
# print(init_pid)

# # two cases on init, pay attention to chips ordering and current_player start index
# # note that init_pid will be whoever starts (SB)
# # should always be call raise or fold

# # {'chips': [1, 2], 'public_card': None, 'hand_cards': ['HQ', 'SJ'], 'current_round': 0, 'current_player': 0, 'legal_actions': ['call', 'raise', 'fold']}
# # {'chips': [2, 1], 'public_card': None, 'hand_cards': ['SQ', 'HQ'], 'current_round': 0, 'current_player': 1, 'legal_actions': ['call', 'raise', 'fold']}

# env.step("call", True)

# state = env.get_perfect_information()
# init_pid = env.get_player_id()

# print("after call", state)

# env.step("check", True)

# # the action that ends preflop betting will always produce public card
# state = env.get_perfect_information()
# print("after check", state)

env.reset()
state = env.get_perfect_information()
init_pid = env.get_player_id()

print(state)
print(init_pid)

env.step("raise", True)

state = env.get_perfect_information()
init_pid = env.get_player_id()

print("after raise", state)

env.step("raise", True)

state = env.get_perfect_information()
init_pid = env.get_player_id()

print("after raise", state)


env.step("call", True)

state = env.get_perfect_information()
init_pid = env.get_player_id()

print("after call", state)
print(env.is_over())
print(env.get_payoffs())

SAMPLE_SIZE = 10


# in leduc, next state is stochastic (going to flop) if we check or if we call and that wasnt a "limp"
def next_state_stochastic(state, action):
    if action == "check":
        return True

    first_call = (state["chips"][0] == 1 and state["chips"][1] == 2) or (
        state["chips"][1] == 1 and state["chips"][0] == 2
    )

    if action == "call" and not first_call:
        return True


def flip(pid):
    return 0 if pid == 1 else 1


def evaluate(state, pid):
    """
    Evaluate a given Leduc Hold'em state from the perspective of the player identified by `pid`.

    This evaluation considers:
      - Hand strength:
          * Preflop (no public card): sum of the hole card values.
          * Postflop (public card revealed): if either hole card matches the public card's rank,
            a pair (strong hand) is assumed (bonus score); otherwise, the value of the highest hole card.
      - Chip advantage: The difference between the player's chips and the average chips across all players.

    Parameters:
      state (dict): The current state of the game with keys such as 'chips', 'public_card', 'hand_cards', etc.
      pid (int): The index of the player for whom the evaluation is performed.

    Returns:
      float: A numerical evaluation score of the state.
    """
    # Retrieve state parameters.
    chips = state.get("chips", [])
    public_card = state.get("public_card")
    hand_cards = state.get("hand_cards", [])

    # Assign numerical values for card ranks.
    # Leduc Hold'em typically uses J, Q, and K.
    ranks = {"J": 11, "Q": 12, "K": 13}

    hand_strength = 0

    # Evaluate hand strength based on whether the public card is present.
    if public_card:
        # Assumption: cards are represented as two-character strings
        # (first character: suit, second character: rank, e.g., 'HK' means Hearts King).
        pub_rank = public_card[1]
        # Check if any of the hole cards forms a pair with the public card.
        has_pair = any(card[1] == pub_rank for card in hand_cards)
        if has_pair:
            # A pair is generally much stronger in Leduc Hold'em.
            hand_strength = 20  # You can tune this bonus as needed.
        else:
            # Without a pair, use the value of the highest hole card.
            hand_strength = max(ranks.get(card[1], 0) for card in hand_cards)
    else:
        # Preflop: simply add the values of the hole cards.
        hand_strength = sum(ranks.get(card[1], 0) for card in hand_cards)

    # Evaluate chip advantage.
    total_chips = sum(chips)
    num_players = len(chips) if chips else 1
    avg_chips = total_chips / num_players
    chip_advantage = chips[pid] - avg_chips

    # Combine the hand strength and chip advantage.
    # The weight for chip advantage is chosen to have an impact while keeping the hand strength primary.
    evaluation_score = hand_strength + 0.1 * chip_advantage

    return evaluation_score


# pid is cur player, bb_id is id of bb player
def expectimax(env, depth, pid, bb_id):
    state = env.get_perfect_information()

    if env.is_over() or depth == 0:
        return evaluate(state, pid)  # heurstic if no exact payoff

    # max case, sb, since we should be calling expectimax from sb perspective to start
    if pid != bb_id:
        bestValue = float("-inf")
        for action in state["legal_actions"]:
            env_copy = copy.deepcopy(env)
            # the reason we do this instead of chance node is because i cannot control the card output as a separate event
            # aka calling step after a "terminal" round action (i.e calling a raise preflop, checking back limp) deals card automatically in env
            if next_state_stochastic(state, action):
                # our chance node case essentially, we cant control the card so well do some sampling and average
                expVal = 0
                for _ in range(SAMPLE_SIZE):
                    env_copy_stochastic = copy.deepcopy(env_copy)
                    env_copy_stochastic.step(action, True)
                    # we want to start the next turn from action on BB (whoever acts second preflop)
                    expVal += expectimax(env_copy_stochastic, depth - 1, bb_id)
                bestValue = max(bestValue, expVal / SAMPLE_SIZE)
            else:
                # normal case, we go to min
                env_copy.step(action, True)
                val = expectimax(env_copy, depth - 1, flip(pid))
                if val > bestValue:
                    bestValue = val
        return bestValue

    # min case, bb, is almost identical w/ obvious changes
    if pid == bb_id:
        worstValue = float("inf")
        for action in state["legal_actions"]:
            env_copy = copy.deepcopy(env)
            if next_state_stochastic(state, action):
                expVal = 0
                for _ in range(SAMPLE_SIZE):
                    env_copy_stochastic = copy.deepcopy(env_copy)
                    env_copy_stochastic.step(action, True)
                    # we want to start the next turn from action on BB (whoever acts second preflop)
                    expVal += expectimax(env_copy_stochastic, depth - 1, bb_id)
                worstValue = min(worstValue, expVal / SAMPLE_SIZE)
            else:
                env_copy.step(action, True)
                val = expectimax(env_copy, depth - 1, flip(pid))
                if val < worstValue:
                    worstValue = val
        return worstValue
