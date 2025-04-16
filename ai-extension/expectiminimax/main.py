import pyspiel
from expectiminimax.algorithms import get_best_action
from expectiminimax.heuristics import heuristic_evaluation_relative


# Load the game and initial state
game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
agent = 0
depth = 10

print("Initial state:")
print(state)

print(state.is_chance_node())
print(state.legal_actions())

# J1
state.apply_action(0)
# K2
state.apply_action(5)

# Compute best action
score, action = get_best_action(state, depth, agent, heuristic_evaluation_relative)
action_str = state.action_to_string(agent, action)
print(f"Best action at depth {depth}: {action_str} (score: {score:.3f})")
