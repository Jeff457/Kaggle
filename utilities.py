from pathlib import Path
import os
import numpy as np

# useful methods go here
# for instance, generating lists of parameter values


def gen_params(start, stop, step):
    """
    :param start: minimum value of parameter
    :param stop: maximum value of parameter
    :param step: amount the generator increments by
    :return: list of parameters within the interval [start, stop)
    """
    return [p for p in np.arange(start, stop, step)]


def report(results, n_top=3):
    """
    From scikit-learn.org
    Print best results from parameter optimization

    :param results: search result from optimize_parameters
    :param n_top: number of results to print
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_file_path(file_name):
    """
    Returns the path to file_name
    :param file_name: file to get the absolute path to
    :return: Path to file_name
    """
    return Path(os.path.dirname(os.path.abspath(__file__)) + file_name)
