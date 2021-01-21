import argparse
import os
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from sympy.ntheory.multinomial import multinomial_coefficients

from logger import RowLogger


def EU(p, N, draws, coeffs=None):
    """
    p: numpy array, a probability distribution over actions
    N: integer, the number of players in the game
    draws: iterable of floats, (averaged) samples from np.random.uniform(0.0, 1.0) defining entries of payoff matrix
    coeffs: list of the items in the dictionary returned by multinomial_coefficients
    """
    assert len(p) >= 1 and N >= 1
    if coeffs is None:
        coeffs = list(multinomial_coefficients(len(p), N).items())

    # TODO vectorize these for-loops, e.g., using scipy.stats.multinomial
    total = 0
    for (coeff, count), draw in zip(coeffs, draws):
        term = count * draw
        for base, power in zip(p, coeff):
            term *= base ** power
        total += term

    return total


def epsilon_vulnerability(A, res, quality, draws, coeffs):
    """
    A: integer, the number of actions available to each player
    res: the result of the symmetric strategy optimization
    quality: expected payoff of symmetric optimum, equal to -res.fun
    draws: iterable of floats, (averaged) samples from np.random.uniform(0.0, 1.0) defining entries of payoff matrix
    coeffs: list of the items in the dictionary returned by multinomial_coefficients
    """
    omitted_indices = np.isclose(res.x, np.zeros(A))  # actions outside the support of res.x
    support_draws = [draw for (coeff, _), draw in zip(coeffs, draws) if
                     np.isclose(np.dot(coeff, omitted_indices), 0)]  # payoffs in the support of res.x
    support_min = min(support_draws)  # minimum over outcomes in the support of res.x
    vulnerability = quality - support_min  # the damage possible by epsilon bribes to pure strategies
    assert vulnerability >= 0, "The expectation should exceed the minimum"

    return vulnerability


def solve(N, A, t, sym):
    """
    N: integer, the number of players in the game
    A: integer, the number of actions available to each player
    t: integer, bookkeeping the current trial
    sym: Boolean, whether or not to average over underlying game matrix
    """
    # ensure that parallel threads are independent trials
    np.random.seed(t)

    # optimize expected utility as function of symmetric strategy probability distribution
    coeffs = list(multinomial_coefficients(A, N).items())
    if sym:
        draws = [np.mean(np.random.uniform(0.0, 1.0, count)) for _, count in coeffs]
    else:
        draws = [np.mean(np.random.uniform(0.0, 1.0, 1)) for _, _ in coeffs]
    with warnings.catch_warnings():
        # suppress trust-constr's note about special case of linear functions
        # warnings.simplefilter("ignore")
        res = minimize(lambda p: -EU(p, N, draws, coeffs=coeffs), np.ones(A) / A,  # method="trust-constr",
                       bounds=[(0, 1)] * A, constraints=({"type": "eq", "fun": lambda x: np.sum(x) - 1}))

    # sense check the optimization result
    # print(res.message)
    # assert res.success
    if not res.success:
        warnings.warn("\nWarning: minimizer failed at N = {}, A = {}, t = {}, sym = {}".format(N, A, t, sym))
        print("res = \n{}".format(res))
    try:
        assert np.isclose(np.sum(res.x), 1.0, atol=1e-02), "x = {0}".format(res.x)
    except:
        print(
            "Warning: throwing away result because optimization solution summed to {:.4f} at N = {}, A = {}, t = {}".format(
                np.sum(res.x), N, A, t))
        return (N, A, t, sym, 0, -np.inf, -np.inf)

    # expected payoff of symmetric optimum
    quality = -res.fun

    # calculate maximum pure strategy payoff
    pure_strategies = np.eye(A)
    pure_payoffs = [EU(p, N, draws, coeffs=coeffs) for p in pure_strategies]

    # if is mixed strategy that beats all pure strategies
    if not np.any(np.isclose(res.x, np.ones(A))) and quality > max(pure_payoffs):
        vulnerability = epsilon_vulnerability(A, res, quality, draws, coeffs)
        return (N, A, t, sym, 1, quality, vulnerability)
    else:
        vulnerability = -np.inf
        return (N, A, t, sym, 0, quality, vulnerability)


def sweep(T, Nmin, Nmax, Amin, Amax, sym, fname, append=False):
    """
    T: integer, number of trials
    Nmin: integer, minimum number of players. trials cover the inclusive range [Nmin, Nmax]
    Nmax: integer, maximum number of players. trials cover the inclusive range [Nmin, Nmax]
    Amin: integer, minimum number of available actions. trials cover the inclusive range [Amin, Amax]
    Amax: integer, maximum number of available actions. trials cover the inclusive range [Amin, Amax]
    sym: Boolean, whether or not to average over underlying game matrix
    fname: string, location to log results
    append: Boolean, whether or not to append to results already at fname
    """
    Ns = range(Nmin, Nmax + 1)
    As = range(Amin, Amax + 1)

    log = RowLogger(fname, columns=["N", "A", "t", "sym", "mixed", "quality", "vulnerability"], append=append)
    arguments = [(N, A, t, sym) for N in Ns for A in As for t in range(T)]
    with ProcessPoolExecutor() as executor:
        for result in executor.map(solve, *zip(*arguments)):
            log.add(*result)


def table(fname, values, pivot=False, to_latex=False):
    root, ext = os.path.splitext(fname)
    assert ext == ".csv", "Input file must be in .csv format"
    out_file = root + "_" + values + ".tex"

    data = pd.read_csv(fname)
    if values == "quality" or values == "vulnerability":
        data = data.loc[data["mixed"] == 1]
        float_format = "{:.3}".format
    elif values == "decrease":
        data = data.loc[data["mixed"] == 1]
        float_format = "{:.1%}".format
    else:
        float_format = "{:.1%}".format
    data = data.groupby(["N", "A"]).aggregate(np.mean).reset_index()
    data["unaffected"] = 1 - data["mixed"]
    data["decrease"] = data["vulnerability"] / data["quality"]
    data = data.pivot(index="N", columns="A", values=values)

    if pivot:
        print(data)
    if to_latex:
        data.to_latex(buf=out_file, float_format=float_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulate solutions to symmetric games")
    parser.add_argument("--players_min", default=2, type=int,
                        help="the fewest number of players in the game. the trials cover the inclusive range [players_min, players_max]")
    parser.add_argument("--players_max", default=5, type=int,
                        help="the largest number of players in the game. the trials cover the inclusive range [players_min, players_max]")
    parser.add_argument("--actions_min", default=2, type=int,
                        help="the fewest number of actions available to each player. the trials cover the inclusive range [actions_min, actions_max]")
    parser.add_argument("--actions_max", default=5, type=int,
                        help="the largest number of actions available to each player. the trials cover the inclusive range [actions_min, actions_max]")
    parser.add_argument("--trials", default=100, type=int, help="the number of trials to run at each point")
    parser.add_argument("--sym", dest="sym", action="store_true",
                        help="symmetrize analysis by averaging over underlying game matrix")
    parser.add_argument("--fname", default="output/results.csv", help="where to save and load experimental results")
    parser.add_argument("--append", dest="append", action="store_true", help="append to results already at fname")
    parser.add_argument("--pivot", dest="pivot", action="store_true", help="display pivot table of results")
    parser.add_argument("--to_latex", dest="to_latex", action="store_true", help="write table of results to .tex file")
    parser.add_argument("--values", default="mixed",
                        help="either 'mixed', 'unaffected', 'quality', 'vulnerability', or 'decrease', whether to fill pivot or to_latex table with the percentage of games whose optimal symmetric strategies are mixed, with the percentage of games whose optimal symmetric strategies are unaffected by payoff perturbation, with the expected payoff of the games whose optimal symmetric strategies are mixed, with how vulnerable the games whose optimal symmetric stragies are mixed are to epsilon bribes, or with the percentage decrease in expected utility caused by epsilon bribes in those games")
    parser.add_argument("--nosim", dest="nosim", action="store_true",
                        help="do not run any simulations but take actions governed by other flags, e.g., pivot")
    parser.set_defaults(sym=False, append=False, pivot=False, to_latex=False, nosim=False)

    args = parser.parse_args()
    if not args.nosim:
        sweep(args.trials, args.players_min, args.players_max, args.actions_min, args.actions_max, args.sym, args.fname,
              append=args.append)

    if args.pivot or args.to_latex:
        table(args.fname, args.values, pivot=args.pivot, to_latex=args.to_latex)
