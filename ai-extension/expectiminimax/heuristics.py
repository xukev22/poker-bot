from utils import calc_hero_equity, simple_strength_heuristic

# Use the same invalid card constant defined in OpenSpiel.
kInvalidCard = -10000


def card_rank(card):
    """
    Converts a card index into a rank value.
    If the card is invalid (not dealt), throws an exception.
    """
    if card == kInvalidCard:
        raise RuntimeError("Card does not exist")
    return card // 2


def h_perfect_info_leduc(state, agent):
    """
    A perfect info heuristic evaluation comparing the agent's score to the opponent's score.

    Preflop:
      Difference between private card ranks.
    Postflop:
      Adds a pair bonus and computes the difference in scores.
    """
    my_card = state.private_card(agent)
    my_rank = card_rank(my_card)

    opponent = 1 - agent
    opp_card = state.private_card(opponent)
    public = state.public_card()

    my_bonus = 10 if (public != kInvalidCard and card_rank(public) == my_rank) else 0
    my_score = my_rank + my_bonus

    if opp_card != kInvalidCard:
        opp_rank = card_rank(opp_card)
        opp_bonus = (
            10 if (public != kInvalidCard and card_rank(public) == opp_rank) else 0
        )
        opp_score = opp_rank + opp_bonus
    else:
        raise ("We should know opponents card")

    return my_score - opp_score


def h_imperfect_info_leduc(state, agent):
    """
    Simple imperfect‑info: assume opp has any card in 0–5
    except your private and the public, uniform EV.
    """
    # 1) your score
    my_card = state.private_card(agent)
    public = state.public_card()
    my_rank = card_rank(my_card)
    my_bonus = 10 if (public != kInvalidCard and card_rank(public) == my_rank) else 0
    my_score = my_rank + my_bonus

    # 2) build unseen pool: cards 0..5 minus your private & the public (if dealt)
    unseen = [c for c in range(6) if c != my_card and c != public]

    # 3) compute all possible opponent scores
    opp_scores = []
    for opp_card in unseen:
        opp_rank = card_rank(opp_card)
        opp_bonus = (
            10 if (public != kInvalidCard and card_rank(public) == opp_rank) else 0
        )
        opp_scores.append(opp_rank + opp_bonus)

    # 4) expected opponent score
    E_opp = sum(opp_scores) / len(opp_scores)

    # 5) return expected difference
    return my_score - E_opp


def h_perfect_info_limit(state, agent):
    # outputs [tie equity?, hero equity, villain equity]
    # given the known board and hole cards
    # print(calc_hero_equity(state, agent))
    tie, this_agent, _ = calc_hero_equity(state, agent)
    return tie + this_agent


def state_pot_size(state, agent):
    """Extract the current pot size (sum of all players' contributions)."""
    game = state.get_game()
    params = game.get_parameters()
    num_players = game.num_players()
    num_suits = params["numSuits"]
    num_ranks = params["numRanks"]
    deck_size = num_suits * num_ranks

    # grab the observation vector for `agent`
    obs = state.observation_tensor(agent)
    offset = num_players + 2 * deck_size
    contribs = obs[offset : offset + num_players]
    return sum(contribs)


def state_contrib(state, agent):
    """How many chips agent has already put into the pot."""
    game = state.get_game()
    params = game.get_parameters()
    num_players = game.num_players()
    deck_size = params["numSuits"] * params["numRanks"]

    obs = state.observation_tensor(agent)
    offset = num_players + 2 * deck_size
    # slice out [spent_by_p0, spent_by_p1, …]
    contribs = obs[offset : offset + num_players]
    return contribs[agent]


def h_perfect_info_weighted_total_limit(state, agent):
    # equity part
    tie, this_eq, _ = calc_hero_equity(state, agent)
    # pot so far
    pot = state_pot_size(state, agent)
    # scale equity by pot
    return (tie + this_eq) * pot


def h_perfect_info_weighted_ctrb_limit(state, agent):
    # equity (ignore ties here if you like)
    _, this_eq, _ = calc_hero_equity(state, agent)
    # pot so far
    pot = state_pot_size(state, agent)
    # what *you* have already invested
    contrib = state_contrib(state, agent)
    # expected net‐EV = equity * pot − your sunk chips
    return this_eq * pot - contrib


def h_imperfect_info_weighted_ctrb_limit(state, agent):
    # equity (ignore ties here if we want)
    this_eq = simple_strength_heuristic(state, agent)
    # pot so far
    pot = state_pot_size(state, agent)
    # what *you* have already invested
    contrib = state_contrib(state, agent)
    # expected net‐EV = equity * pot − your sunk chips
    return this_eq * pot - contrib
