import random


def expectiminimax(state, depth, agent, heursitic_fn, k_samples):
    """
    Recursive expectiminimax search for two-player zero-sum games.

    - state: a PySpiel State object
    - depth: remaining search depth
    - agent: the player id (0 or 1)

    Returns the expectiminimax value for `agent` at `state`.
    """
    print(depth)
    # Terminal or cutoff
    if state.is_terminal() or depth == 0:
        if state.is_terminal():
            return state.returns()[agent]
        return heursitic_fn(state, agent)

    # Chance node
    if state.is_chance_node():
        outcomes = state.chance_outcomes()  # list of (action, prob)
        # if fewer than k, just use them all
        sampled = random.sample(outcomes, min(k_samples, len(outcomes)))
        total_p = sum(p for _, p in sampled)
        value = 0.0

        for action, p in sampled:
            next_state = state.clone()
            next_state.apply_action(action)
            # re‑normalize: p / total_p
            value += (p / total_p) * expectiminimax(
                next_state, depth, agent, heursitic_fn, k_samples
            )
        return value

    # Decision node
    current = state.current_player()
    if current == agent:
        best = float("-inf")
        for action in state.legal_actions():
            nxt = state.clone()
            nxt.apply_action(action)
            val = expectiminimax(nxt, depth - 1, agent, heursitic_fn, k_samples)
            best = max(best, val)
        return best
    else:
        worst = float("inf")
        for action in state.legal_actions():
            nxt = state.clone()
            nxt.apply_action(action)
            val = expectiminimax(nxt, depth - 1, agent, heursitic_fn, k_samples)
            worst = min(worst, val)
        return worst


def get_best_action(state, depth, agent, heursitic_fn, k_samples=10):
    """
    Returns the (value, action) pair with highest expectiminimax score
    for `agent` given `state` and search `depth`.
    """
    best_value = float("-inf")
    best_action = None

    for action in state.legal_actions():
        # print("checking another action")
        nxt = state.clone()
        nxt.apply_action(action)
        val = expectiminimax(nxt, depth - 1, agent, heursitic_fn, k_samples)
        if val > best_value:
            best_value, best_action = val, action
    return best_value, best_action
