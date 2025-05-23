from expectiminimax.heuristics import (
    h_perfect_info_limit,
    h_perfect_info_weighted_ctrb_limit,
    h_perfect_info_weighted_total_limit,
    h_imperfect_info_weighted_ctrb_limit,
)
import pyspiel
from expectiminimax.experiments import run_and_plot_limit

game = pyspiel.load_game(
    "universal_poker("
    "betting=limit,numPlayers=2,numRounds=4,"
    "blind=1 2,raiseSize=2 4 8 16,"
    "maxRaises=3 3 3,"
    "numSuits=4,numRanks=13,"
    "numHoleCards=2,numBoardCards=0 3 1 1)"
)
base_state = game.new_initial_state()
# deal AKs (clubs) to hero
for a in (48, 44):
    base_state.apply_action(a)
# deal two Qs (no club) to villain
for a in (41, 42):
    base_state.apply_action(a)

depth = 4
k_samples = 5
trials = 20

# now loop over heuristics
for fn, name in [
    (h_imperfect_info_weighted_ctrb_limit, "imperf"),
    (h_perfect_info_weighted_ctrb_limit, "ctrb‑weighted"),
    (h_perfect_info_weighted_total_limit, "total‑weighted"),
    (h_perfect_info_limit, "plain‑equity"),
]:
    run_and_plot_limit(
        base_state,
        depth,
        fn,
        name,
        k_samples,
        trials,
        "../ai-extension/graphs",
        "AKs vs QQ",
    )
