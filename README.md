YOU CAN IGNORE EVERYTHING IN AI-EXTENSION/...

vpg/ is our folder containing vanilla policy gradient implementation

`python -m vpg.main` or whatever file you want to run

    - human_play.py, play a human
    - main.py, main training loop
    - mc_models.py, tested them against VPG alongside random agents
    - PolicyNetwork.py, our torch implementation for the MLP

agents.py is for our agents which interface with the given RLCard envs

convergence.py is for creating convergence results of FV vs. EV

examples\_\*.py contain code snippets/light testing (didnt want to delete)

experiments.py is where experiment funcs are stored (create graphs, call our main trainer)

gamma.py is where we learned the effects of a low gamma (see report)

tournament\_\*.py (see model chaining in report)

utils.py is for our state transformer funcs
