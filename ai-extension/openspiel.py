import pyspiel

# Use the same invalid card constant defined in OpenSpiel.
kInvalidCard = -10000


def card_rank(card):
    """
    Converts a card index into a rank value.
    Cards 0,1  → Jack (11)
    Cards 2,3  → Queen (12)
    Cards 4,5  → King (13)
    If the card is invalid (not dealt), returns 0.
    """
    if card == kInvalidCard:
        return 0
    return 11 + (card // 2)


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
        if state.is_terminal():
            # two-player zero-sum game, can use one players return
            return state.returns()[agent]
        else:
            return heuristic_evaluation(state, agent)

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


if __name__ == "__main__":
    game = pyspiel.load_game("leduc_poker")
    state = game.new_initial_state()

    print("Initial state:")
    print(state)

    print(state.is_chance_node())
    print(state.legal_actions())

    state.apply_action(0)
    state.apply_action(2)

    # agent 0 to start
    agent = 0
    depth_limit = 10

    result = expectiminimax(state, depth_limit, agent)
    print(
        "Expectiminimax evaluation for agent {} from the initial state: {}".format(
            agent, result
        )
    )
