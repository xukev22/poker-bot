from utils import calc_hero_equity

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


def h_perfect_info_weighted_total_limit(state, agent):
    # compute old “perfect‐info” number
    tie, this_eq, _ = calc_hero_equity(state, agent)

    # grab the per‐player contributions
    acpc = state.acpc_state()
    pot = sum(acpc.spent)  # total pot so far

    # Option A: just scale equity by pot size
    return (tie + this_eq) * pot


def h_perfect_info_weighted_ctrb_limit(state, agent):
    # compute old “perfect‐info” number
    _, this_eq, _ = calc_hero_equity(state, agent)

    # grab the per‐player contributions
    acpc = state.acpc_state()
    pot = sum(acpc.spent)  # total pot so far
    contrib = acpc.spent[agent]  # how much *you* have put in so far

    # Option B: convert to an *expected net* in chips:
    #    EV = equity * pot − what you’ve already invested
    return this_eq * pot - contrib
