agents.py is for our agents which interact with the given RLCard envs

utils.py is for any useful functions that did not belong in other files

experiments.py is where experiment funcs are stored

main.py is our entry point, where everything ties together. We explore all sorts of different combinations/variations which I will get into later

=======

0. come up with MC implementation, egreedy, training,
   evaluation?

- avoid egreedy (in eval we have predictable agent, may want to use greedy=False)
- how to compare, what metrics?

1. First visit vs. every visit, not noticable difference in leduc (see graphs)

2. The obvious parameters: epsilon and gamma

- epsilon:
  I think lower epsilon allowed for overall better results, see gdocs
- gamma:
  I think lower gamma incentivizes bluffing, see graphs

- Num episodes and update_freq, found that 1m and 100 work well w/ .01
  - could play a role but nothing noticable?, i think big enough to convergence for LEDUC

3. model chaining

- Combinations of agents playing each other
- Combinations of state we care about
- gamma

note: replay buffer? 
note: here i realized a subtle thing, position is important

    Training agent_gen0_aggro_v4 vs. random agent
    Training agent_gen0_std_v4 vs. random agent
    Eval payouts g0av4 vs random: 1.2323 -1.2323
    Eval payouts g0sv4 vs random 1.15085 -1.15085
    (switched) Eval payouts g0av4 vs random: 0.0486 -0.0486
    (switched) Eval payouts g0sv4 vs random 0.1294 -0.1294


