agents.py is for our agents which interact with the given RLCard envs

utils.py is for any useful functions that did not belong in other files

main.py is our entry point, where everything ties together. We explore all sorts of different combinations/variations which I will get into later

The obvious parameters: epsilon and gamma

- epsilon:
  I think lower epsilon
- gamma:
  I think lower gamma incentivizes bluffing

  epsilon annealing? maybe but prob doesnt matter, could play around w/ update freq more

Num episodes and update_freq
- could play a role but nothing noticable?

Combinations of agents playing each other

Combinations of state we care about