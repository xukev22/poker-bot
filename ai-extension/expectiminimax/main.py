from expectiminimax.heuristics import h_perfect_info, h_imperfect_info
from expectiminimax.experiments import run_and_plot


run_and_plot(10, h_perfect_info, heuristic_name="PerfectInfo")
run_and_plot(8, h_perfect_info, heuristic_name="PerfectInfo")
run_and_plot(6, h_perfect_info, heuristic_name="PerfectInfo")
run_and_plot(4, h_perfect_info, heuristic_name="PerfectInfo")
run_and_plot(2, h_perfect_info, heuristic_name="PerfectInfo")

run_and_plot(10, h_imperfect_info, heuristic_name="ImperfectInfo")
run_and_plot(8, h_imperfect_info, heuristic_name="ImperfectInfo")
run_and_plot(6, h_imperfect_info, heuristic_name="ImperfectInfo")
run_and_plot(4, h_imperfect_info, heuristic_name="ImperfectInfo")
run_and_plot(2, h_imperfect_info, heuristic_name="ImperfectInfo")
