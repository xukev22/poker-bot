from expectiminimax.heuristics import h_perfect_info
from expectiminimax.experiments import run_and_plot

run_and_plot(10, h_perfect_info, heuristic_name="PerfectInfo")
