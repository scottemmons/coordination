import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import warnings

import numpy as np
from scipy.optimize import minimize
from sympy.ntheory.multinomial import multinomial_coefficients

from analyze import results_tables
from logger import RowLogger


def get_draws(gamut, coeffs):
    """
    :param gamut: string, the GAMUT game class
    :param coeffs: list, multinomial coefficients corresponding to symmetric game outcomes
    :return: random draws to specify payoffs in the game
    """
    draws = np.random.uniform(0.0, 1.0, len(coeffs))
    if gamut == "RandomGame":
        pass
    elif gamut == "CoordinationGame":
        draws = np.zeros_like(draws)
        for index, (_, count) in enumerate(coeffs):
            if count == 1:  # all players choose same action
                draws[index] = np.random.uniform(0.5, 1.0)
            else:  # at least two different actions are played
                draws[index] = np.random.uniform(0.0, 0.5)
    elif gamut == "CollaborationGame":
        draws = np.zeros_like(draws)
        for index, (_, count) in enumerate(coeffs):
            if count == 1:  # all players choose same action
                draws[index] = 1.0
            else:  # at least two different actions are played
                draws[index] = np.random.uniform(0.0, 199.0 / 200.0)
    else:
        raise ValueError(
            "Must have gamut == 'RandomGame' or gamut == 'CoordinationGame' or gamut == 'CollaborationGame'"
            "but instead have gamut == {}".format(gamut))

    return draws


def uniformly_sample_simplex(A):
    """
    :param A: integer, the number of actions available to each player
    :return: a uniformly random draw over probability distributions with A actions
    """
    distribution = np.random.random(A)
    distribution = -np.log(distribution)
    distribution /= distribution.sum()

    return distribution


def EU(p, N, draws, coeffs=None):
    """
    :param p: numpy array, a probability distribution over actions
    :param N: integer, the number of players in the game
    :param draws: iterable of floats defining entries of game's payoff matrix
    :param coeffs: list, multinomial coefficients corresponding to symmetric game outcomes
    :return: the expected utility of the symmetric strategy profile given by p
    """
    assert len(p) >= 1 and N >= 1
    if coeffs is None:
        coeffs = list(multinomial_coefficients(len(p), N).items())

    # TODO(scottemmons): vectorize these for-loops, e.g., using scipy.stats.multinomial
    total = 0
    for (coeff, count), draw in zip(coeffs, draws):
        term = count * draw
        for base, power in zip(p, coeff):
            term *= base ** power
        total += term

    return total


def EU_individual(index, p, N, draws, coeffs):
    """
    :param index: integer, which action the individual will fix
    :param p: numpy array, a probability distribution over actions for all the other players
    :param N: integer, the number of players in the game
    :param draws: iterable of floats defining entries of game's payoff matrix
    :param coeffs: list, multinomial coefficients corresponding to symmetric game outcomes
    :return: the expected utility when one players plays the action given by `index` and all others play `p`
    """
    draws_individual = []
    coeffs_individual = []
    for (coeff, count), draw in zip(coeffs, draws):
        if coeff[index] == 0:
            continue
        # below is a concrete example where index == 0, coeff == (2, 3, 3), N == 8, and count == \binom{8}{2,3,3}
        # \binom{8}{2,3,3} = 8! / (2! 3! 3!)
        # \binom{7}{1,3,3} = 7! / (1! 3! 3!) = (2 / 8) * \binom{8}{2,3,3}
        coeff_individual = list(coeff)
        coeff_individual[index] -= 1
        coeff_individual = tuple(coeff_individual)
        count_individual = int(round(coeff[index] * count / N))

        draws_individual.append(draw)
        coeffs_individual.append((coeff_individual, count_individual))

    assert len(draws_individual) == len(coeffs_individual) == len(multinomial_coefficients(len(p), N - 1))
    return EU(p, N - 1, draws_individual, coeffs=coeffs_individual)


def get_mixed_vulnerability(N, A, solution, quality, draws, coeffs):
    """
    :param N: integer, the number of players in the game
    :param A: integer, the number of actions available to each player
    :param solution: the output strategy of the symmetric strategy optimization
    :param quality: expected payoff of solution, the symmetric optimum
    :param draws: iterable of floats defining entries of game's payoff matrix
    :param coeffs: list, multinomial coefficients corresponding to symmetric game outcomes
    :return: whether or not the result is a mixed strategy, and if so, its epsilon vulnerability
    """
    pure_payoffs = [EU(pure_strategy, N, draws, coeffs=coeffs) for pure_strategy in np.eye(A)]
    max_pure_payoff = max(pure_payoffs)

    if not np.any(np.isclose(solution, np.ones(A))) and quality > max_pure_payoff:
        vulnerability = epsilon_vulnerability(A, solution, quality, draws, coeffs)
        return 1, vulnerability
    else:
        vulnerability = -np.inf
        return 0, vulnerability


def epsilon_vulnerability(A, solution, quality, draws, coeffs):
    """
    :param A: integer, the number of actions available to each player
    :param solution: the output strategy of the symmetric strategy optimization
    :param quality: expected payoff of solution, the symmetric optimum
    :param draws: iterable of floats defining entries of game's payoff matrix
    :param coeffs: list, multinomial coefficients corresponding to symmetric game outcomes
    :return: the maximum possible loss in expected utility resulting from epsilon bribes to pure strategies
    """
    omitted_indices = np.isclose(solution, np.zeros(A))  # actions outside the support of res.x
    support_draws = [draw for (coeff, _), draw in zip(coeffs, draws) if
                     np.isclose(np.dot(coeff, omitted_indices), 0)]  # payoffs in the support of res.x
    support_min = min(support_draws)  # minimum over outcomes in the support of res.x
    vulnerability = quality - support_min  # the damage possible by epsilon bribes to pure strategies
    assert vulnerability >= 0, "The expectation should exceed the minimum"

    return vulnerability


def global_solution(N, A, t, subtrial, gamut, draws, coeffs):
    """
    :param N: integer, the number of players in the game
    :param A: integer, the number of actions available to each player
    :param t: integer, bookkeeping the current trial
    :param subtrial: integer, bookkeeping the current subtrial
    :param gamut: string, the GAMUT game class
    :param draws: iterable of floats defining entries of game's payoff matrix
    :param coeffs: list, multinomial coefficients corresponding to symmetric game outcomes
    :return: the result of the optimization over symmetric strategies
    """
    # optimize expected utility as function of symmetric strategy probability distribution
    with warnings.catch_warnings():
        # suppress trust-constr's note about special case of linear functions
        # warnings.simplefilter("ignore")
        initialization = uniformly_sample_simplex(A)
        res = minimize(lambda p: -EU(p, N, draws, coeffs=coeffs), initialization,  # method="trust-constr",
                       bounds=[(0, 1)] * A, constraints=({"type": "eq", "fun": lambda x: np.sum(x) - 1}))

    # sense check the optimization result
    # print(res.message)
    # assert res.success
    # TODO(scottemmons): Make edge case where optimization fails compatible with replicator dynamics subtrials
    if not res.success:
        warnings.warn("\nWarning: minimizer failed at N = {}, A = {}, t = {}, gamut = {}".format(N, A, t, gamut))
        print("res = \n{}".format(res))
    if not np.isclose(np.sum(res.x), 1.0, atol=1e-02):
        print(
            "Warning: throwing away result because optimization solution summed to {:.4f} at N = {}, A = {}, t = {}".format(
                np.sum(res.x), N, A, t))
        return N, A, t, subtrial, gamut, "global", 0, -np.inf, -np.inf

    # expected payoff of symmetric optimum
    quality = -res.fun

    # if optimal strategy is mixed, calculate its epsilon vulnerability
    mixed, vulnerability = get_mixed_vulnerability(N, A, res.x, quality, draws, coeffs)
    return N, A, t, subtrial, gamut, "global", mixed, quality, vulnerability


def replicator_dynamics(N, A, t, subtrial, gamut, draws, coeffs, iterations=10000, stepsize=1.):
    """
    :param N: integer, the number of players in the game
    :param A: integer, the number of actions available to each player
    :param t: integer, bookkeeping the current trial
    :param subtrial: integer, bookkeeping the current subtrial
    :param gamut: string, the GAMUT game class
    :param draws: iterable of floats defining entries of game's payoff matrix
    :param coeffs: list, multinomial coefficients corresponding to symmetric game outcomes
    :param iterations: integer, how many steps of the replicator dynamics to run
    :param stepsize: float, coefficient multiplied with the gradient before taking an update step
    :return: the result of the replicator dynamics
    """
    population = uniformly_sample_simplex(A)
    for iteration in range(iterations):
        individual_fitness = np.array([EU_individual(index, population, N, draws, coeffs) for index in range(A)])
        average_fitness = EU(population, N, draws, coeffs=coeffs)
        derivatives = population * (individual_fitness - average_fitness)
        population += stepsize * derivatives
        assert np.all(population >= 0)
        assert np.isclose(population.sum(), 1)

    mixed, vulnerability = get_mixed_vulnerability(N, A, population, average_fitness, draws, coeffs)
    return N, A, t, subtrial, gamut, "replicator", mixed, average_fitness, vulnerability


def solve(N, A, t, subt, gamut):
    """
    :param N: integer, the number of players in the game
    :param A: integer, the number of actions available to each player
    :param t: integer, bookkeeping the current trial
    :param subt: integer, the number of subtrials, i.e., the number of different replicator dynamics initializations
    :param gamut: string, the GAMUT game class
    :return: analysis (to be logged) of the game's solution
    """
    # ensure that parallel threads are independent trials
    np.random.seed(t)

    # results from the global optimization and from the replicator dynamics subtrials
    results = []

    # TODO(scottemmons): double-check that each result is a local optimum in symmetric strategy space?

    # randomly draw a game according to the gamut class
    coeffs = list(multinomial_coefficients(A, N).items())
    draws = get_draws(gamut, coeffs)

    # find the globally optimal solution to the game
    for subtrial in range(subt):
        result = global_solution(N, A, t, subtrial, gamut, draws, coeffs)
        results.append(result)

    # run replicator dynamics to solve the game
    for subtrial in range(subt):
        result = replicator_dynamics(N, A, t, subtrial, gamut, draws, coeffs)
        results.append(result)

    return results


def sweep(T, subt, Nmin, Nmax, Amin, Amax, gamut, fname, append=False):
    """
    Sweep over all experimental parameters and log results to file.

    :param T: integer, number of trials, i.e., number of random games drawn for each game configuration
    :param subt: integer, number of subtrials, i.e., number of different replicator dynamics initializations
    :param Nmin: integer, minimum number of players, trials cover the inclusive range [Nmin, Nmax]
    :param Nmax: integer, maximum number of players. trials cover the inclusive range [Nmin, Nmax]
    :param Amin: integer, minimum number of available actions. trials cover the inclusive range [Amin, Amax]
    :param Amax: integer, maximum number of available actions. trials cover the inclusive range [Amin, Amax]
    :param gamut: Boolean, whether or not to average over underlying game matrix
    :param fname: string, location to log results
    :param append: Boolean, whether or not to append to results already at fname
    """
    Ns = range(Nmin, Nmax + 1)
    As = range(Amin, Amax + 1)

    log = RowLogger(fname,
                    columns=["N", "A", "trial", "subtrial", "gamut", "algorithm", "mixed", "quality", "vulnerability"],
                    append=append)
    arguments = [(N, A, t, subt, gamut) for N in Ns for A in As for t in range(T)]
    with ProcessPoolExecutor() as executor:
        for results in executor.map(solve, *zip(*arguments)):
            for result in results:
                log.add(*result)


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
    parser.add_argument("--trials", default=100, type=int, help="the number of trials to run at each game setting")
    parser.add_argument("--subtrials", default=10, type=int,
                        help="the number of different initializations of each optimization to run")
    parser.add_argument("--gamut", default="RandomGame",
                        choices=["RandomGame", "CoordinationGame", "CollaborationGame"],
                        help="the class of GAMUT games to consider")
    parser.add_argument("--fname", default="output/results.csv", help="where to save and load experimental results")
    parser.add_argument("--append", dest="append", action="store_true", help="append to results already at fname")
    parser.add_argument("--pivot", dest="pivot", action="store_true", help="display pivot table of results")
    parser.add_argument("--to_latex", dest="to_latex", action="store_true", help="write table of results to .tex file")
    parser.add_argument("--values", default="mixed",
                        choices=["mixed", "unaffected", "quality", "vulnerability", "decrease"],
                        help="whether to fill pivot or to_latex table with the percentage of games whose optimal symmetric strategies are mixed (choice == 'mixed'), with the percentage of games whose optimal symmetric strategies are unaffected by payoff perturbation (choice == 'unaffected'), with the expected payoff of the games whose optimal symmetric strategies are mixed (choice == 'quality'), with how vulnerable the games whose optimal symmetric stragies are mixed are to epsilon bribes (choice == 'vulnerability'), or with the percentage decrease in expected utility caused by epsilon bribes in those games (choice == 'decrease')")
    parser.add_argument("--nosim", dest="nosim", action="store_true",
                        help="do not run any simulations but take actions governed by other flags, e.g., pivot")
    parser.set_defaults(append=False, pivot=False, to_latex=False, nosim=False)

    args = parser.parse_args()
    if not args.nosim:
        sweep(args.trials, args.subtrials, args.players_min, args.players_max, args.actions_min, args.actions_max,
              args.gamut, args.fname, append=args.append)

    if args.pivot or args.to_latex:
        outdir, _ = os.path.split(args.fname)
        results_tables(args.fname, outdir, pivot=args.pivot, to_latex=args.to_latex)
