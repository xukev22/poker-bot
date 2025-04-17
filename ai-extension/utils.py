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


import holdem_calc


def state_to_card_info(state):
    # collect only the chance‐node (player == -1) actions in deal order
    cards = [mapping[step.action] for step in state.full_history() if step.player == -1]

    # slice into hero, villain, board
    hero = cards[0:2]
    villain = cards[2:4]
    board = cards[4:]

    return hero, villain, board


def calc_hero_equity(state):
    hero, villain, board = state_to_card_info(state)
    return holdem_calc.calculate(board, True, 1, None, [*hero, *villain], False)
