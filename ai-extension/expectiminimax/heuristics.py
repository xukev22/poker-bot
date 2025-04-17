# from pokerstove import Range, Evaluator

# # define two specific hands (single‐hand ranges)
# hand1 = Range("AsKd")  # Ace♠ King♦
# hand2 = Range("QcJs")  # Queen♣ Jack♠

# board = []  # e.g. preflop
# evaluator = Evaluator()

# # compute exact equity by enumerating all possible 5‑card boards
# eq1, eq2, ties = evaluator.enumerate_equity([hand1], [hand2], board)
# total = eq1 + eq2 + ties
# print(f"Hand1 exact equity: {eq1/total:.2%}")


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


def h_perfect_info(state, agent):
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


def h_imperfect_info(state, agent):
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
