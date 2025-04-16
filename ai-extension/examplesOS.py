import pyspiel
import numpy as np

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
print(state.is_chance_node())
print(state.legal_actions())
# print(state)

outcomes = state.chance_outcomes()
print("Chance outcomes:", outcomes)

state.apply_action(0)

print(state.is_chance_node())
print(state.legal_actions())

state.apply_action(1)

print(state.is_chance_node())
print(state.legal_actions())

print(state)

cur_player = state.current_player()

print("Mapping of legal actions to human-readable descriptions:")
for action in state.legal_actions():
    print(f"Action {action}: {state.action_to_string(cur_player, action)}")
    
    
    
state.apply_action(2)

print(state.is_chance_node())
print(state.legal_actions())

print(state)

cur_player = state.current_player()

print("Mapping of legal actions to human-readable descriptions:")
for action in state.legal_actions():
    print(f"Action {action}: {state.action_to_string(cur_player, action)}")
    
state.apply_action(2)

print(state.is_chance_node())
print(state.legal_actions())

print(state)

cur_player = state.current_player()

print("Mapping of legal actions to human-readable descriptions:")
for action in state.legal_actions():
    print(f"Action {action}: {state.action_to_string(cur_player, action)}")

#     For a chance node:
# The action_id corresponds to the card to be dealt or revealed. The cards are ordered in a fixed way: starting with the lowest card of the first suit, then the lowest card of the second suit, then the next lowest of the first suit, and so on. For example, in the standard two-player version of Leduc Poker (which uses 6 cards), the mapping is:

# Action 0: Jack of the first suit (often denoted as "Jack1")

# Action 1: Jack of the second suit ("Jack2")

# Action 2: Queen of the first suit ("Queen1")

# Action 3: Queen of the second suit ("Queen2")

# Action 4: King of the first suit ("King1")

# Action 5: King of the second suit ("King2")

# This means that when you see the chance outcomes listed as [0, 1, 2, 3, 4, 5], they are encoding which card is being dealt/revealed by following this order.

# while not state.is_terminal():
#     legal_actions = state.legal_actions()
#     if state.is_chance_node():
#         # Sample a chance event outcome.
#         outcomes_with_probs = state.chance_outcomes()
#         action_list, prob_list = zip(*outcomes_with_probs)
#         action = np.random.choice(action_list, p=prob_list)
#         state.apply_action(action)
#     else:
#         # The algorithm can pick an action based on an observation (fully observable
#         # games) or an information state (information available for that player)
#         # We arbitrarily select the first available action as an example.
#         action = legal_actions[0]
#         state.apply_action(action)
