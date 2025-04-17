import holdem_calc


ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
suits = ["c", "d", "h", "s"]
mapping = {i: f"{ranks[i//4]}{suits[i%4]}" for i in range(52)}

# Example:
# mapping[0]  == '2c'
# mapping[1]  == '2d'
# mapping[2]  == '2h'
# mapping[3]  == '2s'
# ...
# mapping[32] == 'Tc'
# mapping[44] == 'Kc'
# mapping[48] == 'Ac'


def state_to_card_info(state):
    # collect only the chance‐node (player == -1) actions in deal order
    cards = [mapping[step.action] for step in state.full_history() if step.player == -1]

    # slice into hero, villain, board
    hero = cards[0:2]
    villain = cards[2:4]
    board = cards[4:]

    return hero, villain, board


# 0 assumed to be hero, aka first two hole cards
def calc_hero_equity(state, agent):
    hero, villain, board = state_to_card_info(state)
    hole_cards = [*hero, *villain] if agent == 0 else [*villain, *hero]
    return holdem_calc.calculate(board, True, 1, None, hole_cards, False)


from collections import Counter

_rank_map = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def _card_rank(card):
    return _rank_map[card[0]]


def _card_suit(card):
    return card[1]


def simple_strength_heuristic(state, agent):
    """
    A very cheap [0..1] estimate of how good our hand is,
    *without* ever iterating over villain hole cards.

    Features:
      - Pair / trips with the board
      - Flush‐draw potential
      - Straight‐draw potential
      - High‐card strength
    """
    hero, _, board = state_to_card_info(state)
    hero_cards = hero if agent == 0 else state_to_card_info(state)[1]
    cards = hero_cards + board

    # count ranks & suits
    ranks = [_card_rank(c) for c in cards]
    suits = [_card_suit(c) for c in cards]
    rc = Counter(ranks)
    sc = Counter(suits)

    score = 0.0

    # 1) Pair/trips bonus
    for r in (_card_rank(c) for c in hero_cards):
        if rc[r] >= 2:  # we’ve paired the board
            score += 0.25 * (rc[r] - 1)

    # 2) Flush‐draw bonus (need 4 cards of same suit total)
    for s in set(_card_suit(c) for c in hero_cards):
        if sc[s] >= 4:
            score += 0.2

    # 3) Straight‐draw bonus: look for any run of 4 distinct ranks
    uniq = sorted(set(ranks))
    max_run = 1
    cur = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i - 1] + 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    if max_run >= 4:
        score += 0.2

    # 4) High‐card kicker: highest hole‐card rank scaled into [0..0.3]
    hv = max(_card_rank(c) for c in hero_cards)
    score += 0.3 * ((hv - 2) / (14 - 2))

    # clamp to [0,1]
    return min(score, 1.0)
