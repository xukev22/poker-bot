from expectiminimax.heuristics import (
    h_perfect_info_leduc,
    h_imperfect_info_leduc,
)

from expectiminimax.experiments import run_and_plot_leduc


run_and_plot_leduc(8, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot_leduc(6, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot_leduc(4, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot_leduc(2, h_perfect_info_leduc, heuristic_name="PerfectInfo")

run_and_plot_leduc(8, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot_leduc(6, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot_leduc(4, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot_leduc(2, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
