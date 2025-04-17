from expectiminimax.heuristics import (
    h_perfect_info_limit,
    h_perfect_info_weighted_ctrb_limit,
    h_perfect_info_weighted_total_limit,
)
from expectiminimax.algorithms import get_best_action
import pyspiel

game = pyspiel.load_game(
    "universal_poker("
    "betting=limit,numPlayers=2,numRounds=4,"
    "blind=1 2,raiseSize=2 4 8 16,"
    "maxRaises=3 3 3,"
    "numSuits=4,numRanks=13,"
    "numHoleCards=2,numBoardCards=0 3 1 1)"
)
state = game.new_initial_state()

# deal two As
state.apply_action(48)
state.apply_action(49)

# deal two 5s to other player
state.apply_action(13)
state.apply_action(14)

depth = 4
agent = 0
# heuristic_fn = h_perfect_info_weighted_ctrb_limit
heuristic_fn = h_perfect_info_weighted_total_limit

score, action = get_best_action(state, depth, agent, heuristic_fn, k_samples=5)
action_str = state.action_to_string(agent, action)

print(state)
print(f"score: {score}, best action: {action} aka {action_str}")
