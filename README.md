# Symmetry, Equilibria, and Robustness in Common-Payoff Games

Code to reproduce the simulated experiments in the paper "Symmetry, Equilibria, and Robustness in Common-Payoff Games."

## To install:
`pip install .`

## To launch the experiments:
The following command will run the simulated experiments in the paper:

`python sim.py --gamut RandomGame --players_min 2 --players_max 5 --actions_min 2 --actions_max 5 --trials 100 --subtrials 10 --pivot --to_latex`

The above command will draw `RandomGame`s. To draw `CoordinationGame`s or `CollaborationGame`s, modify the `--gamut` flag accordingly.
