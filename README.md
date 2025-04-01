My high-level design plan (may change):

poker-bot-project/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py         # Abstract or base class for agents
│   ├── mc_agent.py           # Implementation of Monte Carlo based agents
│   ├── td0_agent.py          # Implementation of TD(0) agents (if applicable)
│   └── fixed_agent.py        # A fixed strategy agent (e.g., RandomAgent)
│
├── models/
│   ├── __init__.py
│   ├── leduc_model.py        # Model definitions for Leduc Hold'em, etc.
│   └── custom_model.py       # Other model variants
│
├── experiments/
│   ├── __init__.py
│   ├── experiment_runner.py  # Script to set up, run, and manage experiments
│   ├── pairing_schemes.py    # Functions or classes to handle different match-ups (MC vs fixed, MC vs MC, etc.)
│   └── config.yaml           # Config file to specify experiment parameters
│
├── evaluation/
│   ├── __init__.py
│   ├── result_parser.py      # Parse raw results and log data
│   └── visualization.py      # Functions to generate plots and statistics
│
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Custom logging functions or setup
│   └── helpers.py            # Utility functions (e.g., seed setting, configuration parsing)
│
└── main.py                   # Entry point to kick off experiments
