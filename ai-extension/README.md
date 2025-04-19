## Quickstart

`cd ai-extension`
// in ai-extension folder run:
`python -m expectiminimax.limit_main` or whatever main file you want to run 

for CS4100, all code is in the ai-extension/ folder, EVERYTHING OUTSIDE OF THIS IS IRRELEVANT FOR THIS CLASS
graphs are all figures in report + extra figures

examples/_ is what I used for testing/code snippets
expectiminimax/_ is where my core code is

- algorithms.py: includes expectiminimax outline + best action helper
- experiments.py: runs experiments w/ parameters on expectiminimax
- heuristics.py: (s,a) -> value estimates
- leduc_main.py: my main runner for leduc variant
- limit_imperfect_main.py/limit_main.py: my main runners for limit variant

utils.py: helper funcs
