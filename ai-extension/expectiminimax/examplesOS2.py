import pyspiel

# Use the same invalid card constant defined in OpenSpiel.
kInvalidCard = -10000


def card_rank(card):
    """
    Converts a card index into a rank value.
    If the card is invalid (not dealt), throw exception.
    """
    if card == kInvalidCard:
        raise RuntimeError("Card does not exist")
    return card // 2


def heuristic_evaluation_relative(state, agent):
    """
    A relative heuristic evaluation function for Leduc Poker.

    This function compares the agent's hand against the opponent's hand.
    Assumes there are two players (agents 0 and 1) and that the opponent's private card
    is accessible (e.g. in simulation or after showdown).

    Preflop:
      The evaluation uses the private card ranks (difference between agent’s and opponent’s).

    Postflop:
      A bonus is awarded for a pair (private card matching the public card),
      and the final score is the difference between the agent's total and the opponent's.
    """
    # Get the agent's private card and compute its rank.
    my_card = state.private_card(agent)
    my_rank = card_rank(my_card)

    # Identify the opponent assuming a two-player game.
    opponent = 1 - agent
    opp_card = state.private_card(opponent)

    # Evaluate bonus if the public card forms a pair.
    public = state.public_card()
    my_bonus = 10 if (public != kInvalidCard and card_rank(public) == my_rank) else 0
    my_score = my_rank + my_bonus

    # Check if the opponent's card is known.
    if opp_card != kInvalidCard:
        opp_rank = card_rank(opp_card)
        opp_bonus = (
            10 if (public != kInvalidCard and card_rank(public) == opp_rank) else 0
        )
        opp_score = opp_rank + opp_bonus
    else:
        # Fallback: If the opponent’s card is hidden, one common choice is to assume an average value.
        # For a deck like in Leduc (typically two copies each of three ranks) the average rank might be:
        # (sum of all possible card ranks minus the agent’s card) / (number of remaining cards).
        # This is a simplified placeholder; in a full implementation you might enumerate the remaining cards.
        average_rank = 2  # Example: if ranks are 1, 2, 3 then the average is (1+3)/2 = 2 when excluding your card.
        opp_score = average_rank

    return my_score - opp_score


def heuristic_evaluation(state, agent):
    """
    A simplified heuristic evaluation function for Leduc Poker that ignores chip counts.

    Preflop:
      Uses the rank of the agent's private card.

    Postflop:
      Checks if the agent's private card matches the public card to form a pair.
      If a pair is made, a bonus is awarded; otherwise, only the private card rank is used.
    """
    # Get the agent's private card.
    my_card = state.private_card(agent)
    private_rank = card_rank(my_card)

    # Get the public card (if any).
    public = state.public_card()

    if public == kInvalidCard:
        # Preflop: Only the private card determines the hand strength.
        return private_rank
    else:
        public_rank = card_rank(public)
        # If the public card matches the private card rank, award a bonus for a pair.
        if public_rank == private_rank:
            return private_rank + 10
        else:
            return private_rank


# agent is 0 or 1
# we terminate at d = 0
# state must be deep copied to explore routes
def expectiminimax(state, depth, agent):
    # print(depth, agent)
    # terminal state or depth limit reached
    if state.is_terminal() or depth == 0:
        if depth == 0:
            print("depth 0 reached")
        if state.is_terminal():
            # two-player zero-sum game, can use one players return
            return state.returns()[agent]
        else:
            return heuristic_evaluation_relative(state, agent)

    # chance node: calculate the expected value over all outcomes
    if state.is_chance_node():
        value = 0
        for action, prob in state.chance_outcomes():
            next_state = state.clone()
            next_state.apply_action(action)
            value += prob * expectiminimax(next_state, depth - 1, agent)
        return value

    # decision node: determine if this is a max node or a min node
    current_player = state.current_player()

    # max node: agents turn
    if current_player == agent:
        best_value = float("-inf")
        for action in state.legal_actions():
            next_state = state.clone()
            next_state.apply_action(action)
            value = expectiminimax(next_state, depth - 1, agent)
            best_value = max(best_value, value)
        return best_value
    # min node: opponents turn
    else:
        worst_value = float("inf")
        for action in state.legal_actions():
            next_state = state.clone()
            next_state.apply_action(action)
            value = expectiminimax(next_state, depth - 1, agent)
            worst_value = min(worst_value, value)
        return worst_value


def get_best_action(state, depth, agent):
    best_value = float("-inf")
    best_action = None

    # consider each legal action
    for action in state.legal_actions():
        # dont mutate original
        next_state = state.clone()
        next_state.apply_action(action)
        # recurse
        value = expectiminimax(next_state, depth - 1, agent)

        if value > best_value:
            best_value = value
            best_action = action

    return best_value, best_action


if __name__ == "__main__":
    game = pyspiel.load_game("leduc_poker")
    state = game.new_initial_state()

    print("Initial state:")
    print(state)

    print(state.is_chance_node())
    print(state.legal_actions())

    # J1
    state.apply_action(4)
    # K1
    state.apply_action(5)

    # agent 0 to start
    agent = 0
    depth_limit = 6

    result, act = get_best_action(state, depth_limit, agent)

    action = state.action_to_string(agent, act)

    print(
        "Expectiminimax evaluation for agent {} from the initial state: {}\nBest action: {}".format(
            agent, result, action
        )
    )
