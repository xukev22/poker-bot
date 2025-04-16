def expectiminimax(state, depth, agent, heursitic_fn):
    """
    Recursive expectiminimax search for two-player zero-sum games.

    - state: a PySpiel State object
    - depth: remaining search depth
    - agent: the player id (0 or 1)

    Returns the expectiminimax value for `agent` at `state`.
    """
    # Terminal or cutoff
    if state.is_terminal() or depth == 0:
        if state.is_terminal():
            return state.returns()[agent]
        return heursitic_fn(state, agent)

    # Chance node
    if state.is_chance_node():
        value = 0.0
        for action, prob in state.chance_outcomes():
            next_state = state.clone()
            next_state.apply_action(action)
            value += prob * expectiminimax(next_state, depth - 1, agent, heursitic_fn)
        return value

    # Decision node
    current = state.current_player()
    if current == agent:
        best = float("-inf")
        for action in state.legal_actions():
            nxt = state.clone()
            nxt.apply_action(action)
            val = expectiminimax(nxt, depth - 1, agent, heursitic_fn)
            best = max(best, val)
        return best
    else:
        worst = float("inf")
        for action in state.legal_actions():
            nxt = state.clone()
            nxt.apply_action(action)
            val = expectiminimax(nxt, depth - 1, agent, heursitic_fn)
            worst = min(worst, val)
        return worst


def get_best_action(state, depth, agent, heursitic_fn):
    """
    Returns the (value, action) pair with highest expectiminimax score
    for `agent` given `state` and search `depth`.
    """
    best_value = float("-inf")
    best_action = None

    for action in state.legal_actions():
        nxt = state.clone()
        nxt.apply_action(action)
        val = expectiminimax(nxt, depth - 1, agent, heursitic_fn)
        if val > best_value:
            best_value, best_action = val, action
    return best_value, best_action
