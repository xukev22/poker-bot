from expectiminimax.heuristics import (
    h_perfect_info_leduc,
    h_imperfect_info_leduc,
    h_perfect_info_limit,
)
from expectiminimax.algorithms import get_best_action
import pyspiel


from expectiminimax.experiments import run_and_plot_leduc

game = pyspiel.load_game(
    "universal_poker("
    "betting=limit,numPlayers=2,numRounds=4,"
    "blind=1 2,raiseSize=2 4 8 16,"
    "maxRaises=3 3 3,"
    "numSuits=4,numRanks=13,"
    "numHoleCards=2,numBoardCards=0 3 1 1)"
)
state = game.new_initial_state()

# deal two 3s
state.apply_action(6)
state.apply_action(7)

# deal two 5s to other player
state.apply_action(13)
state.apply_action(14)

depth = 3
agent = 0
heuristic_fn = h_perfect_info_limit

score, action = get_best_action(state, depth, agent, heuristic_fn)
action_str = state.action_to_string(agent, action)

print(f"score: {score}, best action: {action} aka {action_str}")

run_and_plot_leduc(8, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot_leduc(6, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot_leduc(4, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot_leduc(2, h_perfect_info_leduc, heuristic_name="PerfectInfo")

run_and_plot_leduc(8, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot_leduc(6, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot_leduc(4, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot_leduc(2, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
