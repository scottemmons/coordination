# For Learning in Symmetric Teams, Local Optima are Global Nash Equilibria

Code to reproduce the simulated experiments in the paper "For Learning in Symmetric Teams, Local Optima are Global Nash Equilibria"

## To install:
`pip install .`

## To launch the experiments:
The following command will run the simulated experiments in the paper:

`python sim.py --gamut RandomGame --players_min 2 --players_max 5 --actions_min 2 --actions_max 5 --trials 100 --subtrials 10 --pivot --to_latex`

The above command will draw `RandomGame`s. To draw `CoordinationGame`s or `CollaborationGame`s, modify the `--gamut` flag accordingly.
