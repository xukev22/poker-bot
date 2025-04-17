import pyspiel

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
    A relative heuristic evaluation comparing the agent's score to the opponent's score.

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
        print("perfect info")
    else:
        raise ("We should know opponents card")

    return my_score - opp_score
