YOU CAN IGNORE EVERYTHING IN AI-EXTENSION/...

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
- Combinations of state we care about (did not do but considered)
- gamma

note: replay buffer?
note: here i realized a subtle thing, position is important

    Training agent_gen0_aggro_v4 vs. random agent
    Training agent_gen0_std_v4 vs. random agent
    Eval payouts g0av4 vs random: 1.2323 -1.2323
    Eval payouts g0sv4 vs random 1.15085 -1.15085
    (switched) Eval payouts g0av4 vs random: 0.0486 -0.0486
    (switched) Eval payouts g0sv4 vs random 0.1294 -0.1294

Training agent_gen1_aggro_v4 vs. agent_gen1_std_v4 agent
Eval payouts g1av4 vs random: 0.95405 -0.95405
Eval payouts g1sv4 vs random: 1.1292 -1.1292
Saving gen 1 v4 agents
Eval payouts g1av4 vs g1sv4: -0.6478 0.6478

Eval payouts g2av4 vs random: 0.9761 -0.9761
Eval payouts g2sv4 vs random: 1.1596 -1.1596
Saving gen 2 v4 agents
Eval payouts g2av4 vs g2sv4: -0.6285 0.6285

note: training not improving, realized that replay buffer probably needed, also that we are just overfitting to whoever we play against

4. extending to limit holdem (to look for improvements)

note: poor initial results
Training agent_ev4 vs. random agent
0.20605 -0.20605

Ways to shrink your pickle
Simplify your state representation
Switch back to process_leduc_state_v1 (hand, public_card, chip counts) or even v2. That slashes the number of distinct states by orders of magnitude.

Prune low‑count entries before saving

# e.g. drop any (s,a) seen fewer than 5 times
for s in list(agent.N):
    for a in list(agent.N[s]):
        if agent.N[s][a] < 5:
            del agent.Q[s][a]
            del agent.N[s][a]
    if not agent.N[s]:
        del agent.N[s]
        del agent.Q[s]
That removes the tail of your “long‑tail” state‑action pairs.

Customize what gets pickled
Implement __getstate__/__setstate__ on your agent class to only serialize Q (or even a filtered version of it), skipping N or any other ephemeral data.

Training agent_ev4 vs. random agent
0.180815 -0.180815

5. vanilla policy gradient
