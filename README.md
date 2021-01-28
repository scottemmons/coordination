# Symmetry, Equilibria, and Robustness in Common-Payoff Games

Code to reproduce the simulated experiments in the paper "Symmetry, Equilibria, and Robustness in Common-Payoff Games."

## To install:
`pip install .`

## To launch the experiments:
The following command will run the simulated experiments in the paper:

`python sim.py --players_min 2 --players_max 5 --actions_min 2 --actions_max 5 --trials 100000 --pivot --to_latex --values mixed`

The above command will draw games according to the _symmetric measure_. To draw games according to the _symmetrized measure_, add the `--sym` flag.

Additionally, the above command shows the fraction of games who global symmetric optima are not local optima in possibly-asymmetric strategy space. To see the average decrease in expected utility that worst-case infinitesimal asymmetric payoff perturbations cause, replace `--values unaffected` with `--values decrease`.