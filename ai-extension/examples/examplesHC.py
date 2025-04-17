import pyspiel

# game = pyspiel.load_game("leduc_poker")
# state = game.new_initial_state()
# state.apply_action(0)
# state.apply_action(1)
# print(state.is_chance_node())
# print(state.legal_actions())
# print(state)
# print("Mapping of legal actions to human-readable descriptions:")
# for action in state.legal_actions():
#     print(f"Action {action}: {state.action_to_string(0, action)}")
# state.apply_action(1)

# print(state)
# state.apply_action(2)

# print(state.legal_actions())

# state.apply_action(2)
# state.apply_action(1)


# print(state.legal_actions())

# state.apply_action(2)

# print(state)
# print(state.legal_actions())


# print(state)
# print("Mapping of legal actions to human-readable descriptions:")
# for action in state.legal_actions():
#     print(f"Action {action}: {state.action_to_string(0, action)}")

# 3 bet max, heads up limit holdem, w/ standard deck, standard betting rounds
game = pyspiel.load_game(
    "universal_poker("
    "betting=limit,numPlayers=2,numRounds=4,"
    "blind=1 2,raiseSize=2 4 8 16,"
    "maxRaises=3 3 3,"
    "numSuits=4,numRanks=13,"
    "numHoleCards=2,numBoardCards=0 3 1 1)"
)
state = game.new_initial_state()
print(len(state.legal_actions()))  # should be 52

# references
# https://github.com/crissilvaeng/acpc-server
# https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/canonical_game_strings.cc

from utils import state_to_card_info, calc_hero_equity

# should be true and 0-51 cards to deal
print(state.is_chance_node())
print(state.legal_actions())

print(state.chance_outcomes())

# deal two 3s
state.apply_action(6)
state.apply_action(7)
# state.apply_action(48)

# still true and cards to deal
print(state.is_chance_node())
print(state.legal_actions())

# deal two 5s to other player
state.apply_action(13)
state.apply_action(14)

# false now, say some betting occurs
print(state.is_chance_node())
print(state.legal_actions())

print(state)

# raise
state.apply_action(2)


print(state)

# raise back, then we call
state.apply_action(2)
state.apply_action(1)

print(state)

# back to true, remaining cards are legal acts
print(state.is_chance_node())
print(state.legal_actions())

print("hiya", calc_hero_equity(state, 0))


# deal 4c,3c,2c
print(state.apply_action(0))
print(state.apply_action(4))
print(state.apply_action(8))


print(state)

# now up to p0 to act (can check/call or bet)
print(state.is_chance_node())
print(state.legal_actions())


import holdem_calc

# import parallel_holdem_calc

# The first element in the list corresponds to the probability that a tie takes place. Each element after that corresponds to the probability one of the hole cards the user provides wins the hand. These probabilities occur in the order in which you list them.
# https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h
for pair in state.full_history():
    print(pair.player, pair.action)

print(state_to_card_info(state))

print(
    holdem_calc.calculate(
        ["As", "Ks", "Jd"], True, 1, None, ["8s", "7s", "Qc", "Th"], False
    )
)

print(calc_hero_equity(state, 1))
# print parallel_holdem_calc.calculate(None, True, 1, None, ["8s", "7s", "Ad", "Ac"], False)
