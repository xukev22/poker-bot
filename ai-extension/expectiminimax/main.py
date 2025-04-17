from expectiminimax.heuristics import h_perfect_info_leduc, h_imperfect_info_leduc
from expectiminimax.experiments import run_and_plot


run_and_plot(10, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot(8, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot(6, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot(4, h_perfect_info_leduc, heuristic_name="PerfectInfo")
run_and_plot(2, h_perfect_info_leduc, heuristic_name="PerfectInfo")

run_and_plot(10, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot(8, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot(6, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot(4, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
run_and_plot(2, h_imperfect_info_leduc, heuristic_name="ImperfectInfo")
