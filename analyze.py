import argparse
import os

import pandas as pd


def validate_and_load(input_file):
    """
    :param input_file: string, filename from which experimental results will be loaded
    :return: DataFrame, the data in `input_file`
    """
    _, tail = os.path.split(input_file)
    prefix, ext = os.path.splitext(tail)
    assert ext == ".csv", "Input file must be in .csv format but instead is {} format".format(ext)

    return pd.read_csv(input_file), prefix


def get_max_qualities(data):
    """
    :param data: DataFrame, loaded experimental results
    :return: DataFrame, containing for each trial only the subtrial with maximum quality
    """
    global_ = data.loc[data["algorithm"] == "global"]
    replicator = data.loc[data["algorithm"] == "replicator"]
    global_max = global_.groupby(["N", "A", "trial"]).max()
    replicator_max = replicator.groupby(["N", "A", "trial"]).max()
    max_qualities = global_max.where(global_max["quality"] >= replicator_max["quality"], replicator_max)

    return max_qualities


def pivot_and_save(table, pivot, to_latex, pivot_text="", output_file=None, float_format_str="{:.3}"):
    """
    Print and/or save a table of `data`.

    :param table: DataFrame, the table to be printed and / or saved
    :param pivot: Boolean, if True will print table of `data`
    :param to_latex: Boolean, if True will write table of `data` to .tex file
    :param pivot_text: string, if `pivot` then `pivot_text` will be printed before the pivot table
    :param output_file: string, filename where output table will be written
    :param float_format_str: string, specifying the float format for the .tex file
    """
    if to_latex and output_file is None:
        raise ValueError("Must provide an output_file to save data to_latex")

    if pivot:
        print(pivot_text)
        print(table)
    if to_latex:
        table.to_latex(buf=output_file, float_format=float_format_str.format)


def max_mixed_table(data):
    """
    :param data: DataFrame, containing all experimental results
    :return: DataFrame, showing in what fraction of trials the best solution we find is mixed
    """
    max_qualities = get_max_qualities(data)
    table = max_qualities.pivot_table(values="mixed", index="N", columns="A", aggfunc="mean")

    pivot_text = "\nFraction of trials for which the best solution we find is mixed:"
    return table, "max_mixed", "{:.1%}", pivot_text


def max_decrease_table(data):
    """
    :param data: DataFrame, containing all experimental results
    :return: DataFrame, showing average percentage decrease in EU epsilon bribes cause when our best solution is mixed
    """
    max_qualities = get_max_qualities(data)
    max_qualities = max_qualities.loc[max_qualities["mixed"] == 1]
    max_qualities["decrease"] = max_qualities["vulnerability"] / max_qualities["quality"]
    table = max_qualities.pivot_table(values="decrease", index="N", columns="A", aggfunc="mean")

    pivot_text = "\nAverage percentage decrease in EU epsilon bribes cause when the best solution we find is mixed:"
    return table, "max_decrease", "{:.1%}", pivot_text


def replicator_mixed_table(data):
    """
    :param data: DataFrame, containing all experimental results
    :return: DataFrame, showing fraction of replicator dynamics subtrials that are mixed
    """
    replicator = data.loc[data["algorithm"] == "replicator"]
    table = replicator.pivot_table(values="mixed", index="N", columns="A", aggfunc="mean")

    pivot_text = "\nFraction of replicator dynamics subtrials that are mixed:"
    return table, "replicator_mixed", "{:.1%}", pivot_text


def replicator_one_run_optimality(data, eps=1e-3):
    """
    :param data: DataFrame, containing all experimental results
    :param eps: float, epsilon tolerance when comparing two expected utilities
    :return: DataFrame, showing what fraction of single replicator dynamics subtrials have maximum quality among all
              other global and replicator subtrials
    """
    replicator = data.loc[data["algorithm"] == "replicator"]
    max_qualities = get_max_qualities(data)

    for iteration, subtrial in enumerate(replicator["subtrial"].unique()):
        new_max = max_qualities.copy()
        new_max["subtrial"] = subtrial
        if iteration == 0:
            max_subtrials = new_max
        else:
            max_subtrials = max_subtrials.append(new_max)

    max_subtrials = max_subtrials.reset_index().set_index(["N", "A", "trial", "subtrial"]).sort_index()
    replicator_subtrials = replicator.set_index(["N", "A", "trial", "subtrial"]).sort_index()
    max_subtrials["quality"] -= eps  # preparing for float comparison
    replicator_optimal = replicator_subtrials > max_subtrials
    table = replicator_optimal.pivot_table(values="quality", index="N", columns="A", aggfunc="mean")

    pivot_text = "\nFraction of single replicator dynamics subtrials that have maximum quality among all other global " \
                 "and replicator subtrials:"
    return table, "replicator_one_run_optimality", "{:.1%}", pivot_text


def replicator_many_run_optimality(data, eps=1e-3):
    """
    :param data: DataFrame, containing all experimental results
    :param eps: float, epsilon tolerance when comparing two expected utilities
    :return: DataFrame, showing what fraction of trials have at least one replicator dynamics subtrial achieving maximum
              quality among all other global and replicator subtrials
    """
    replicator = data.loc[data["algorithm"] == "replicator"]
    replicator_max = replicator.groupby(["N", "A", "trial"]).max()
    max_qualities = get_max_qualities(data)

    replicator_max.reset_index().set_index(["N", "A", "trial"]).sort_index()
    max_qualities.reset_index().set_index(["N", "A", "trial"]).sort_index()
    max_qualities["quality"] -= eps  # preparing for float comparison
    replicator_optimal = replicator_max > max_qualities
    table = replicator_optimal.pivot_table(values="quality", index="N", columns="A", aggfunc="mean")

    subtrials = len(replicator["subtrial"].unique())
    pivot_text = "\nFraction of trials for which at least one of {} replicator dynamics subtrials achieves maximum " \
                 "quality among all global and replicator subtrials:".format(subtrials)
    return table, "replicator_{}_run_optimality".format(subtrials), "{:.1%}", pivot_text


def table_template(input_file, table_function, pivot=False, to_latex=True, outdir=None):
    """
    Process the data in `input_file` according to `table_function`. Then print results and/or save results to file.

    :param input_file: string, filename from which experimental results will be loaded
    :param table_function: function, how to process the data in the input_file
    :param pivot: Boolean, if True will print table
    :param to_latex: Boolean, if True will write table to .tex file
    :param outdir: string, directory where table will be saved
    """
    if to_latex and outdir is None:
        raise ValueError("Must provide an outdir to save data to_latex")

    data, prefix = validate_and_load(input_file)
    table, file_tag, float_format_str, pivot_text = table_function(data)

    if outdir is None:
        output_file = None
    else:
        output_file = os.path.join(outdir, "{}_{}.tex".format(prefix, file_tag))

    pivot_and_save(table, pivot, to_latex, pivot_text=pivot_text, output_file=output_file,
                   float_format_str=float_format_str)


def results_tables(fname, outdir, pivot=False, to_latex=True):
    """
    Create tables that give summary statistics of the results.

    :param fname: string, filename from which experimental results will be loaded
    :param outdir: string, directory where results tables will be saved
    :param pivot: Boolean, if True will print table
    :param to_latex: Boolean, if True will write table to .tex file
    """
    root, ext = os.path.splitext(fname)
    assert ext == ".csv", "Input file must be in .csv format but instead is {} format".format(ext)

    table_template(fname, max_mixed_table, pivot=pivot, to_latex=to_latex, outdir=outdir)
    table_template(fname, max_decrease_table, pivot=pivot, to_latex=to_latex, outdir=outdir)
    table_template(fname, replicator_mixed_table, pivot=pivot, to_latex=to_latex, outdir=outdir)
    table_template(fname, replicator_one_run_optimality, pivot=pivot, to_latex=to_latex, outdir=outdir)
    table_template(fname, replicator_many_run_optimality, pivot=pivot, to_latex=to_latex, outdir=outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyze simulation results")
    parser.add_argument("--fname", default="output/results.csv",
                        help="filename from which experimental results will be loaded")
    parser.add_argument("--outdir", default="output", help="directory where results tables will be saved")
    parser.add_argument("--pivot", dest="pivot", action="store_true", help="display pivot table of results")
    parser.add_argument("--no_latex", dest="no_latex", action="store_true",
                        help="do *not* write table of results to .tex file")
    parser.set_defaults(pivot=False, no_latex=False)

    args = parser.parse_args()
    results_tables(args.fname, args.outdir, pivot=args.pivot, to_latex=not args.no_latex)
