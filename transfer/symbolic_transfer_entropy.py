import numpy as np
from itertools import permutations
from math import log2


"""
symbolic_transfer_entropy.py is a script to calculate the Symbolic Transfer Entropy (STE) between two random variables. 
The limiting dimension is the symbol window length (embedding dimension) because the computation for requires 
an integer array of size w! x w! x w!

Sagar Kumar, 2023
"""


class Symbol:

    def __init__(self,
                 w: int):

        """
        A Symbol is a container for the ensemble of possible permutations of window length w

        :param w: (aka embedding dimension) Symbol window length. Leads to ensemble size of w! and
        w-tuple of symbols. e.g. w=3 => {(3,2,1), (2,1,3), ...}
        """

        self.ensemble: list[tuple[int]] = list(permutations(range(1,w+1)))
        self.labels: dict[tuple[int], int] = {v: n for n,v in enumerate(self.ensemble)}


def symbolize(x: list[float],
              w: int,
              s: int = 1
              ) -> list[tuple[int]]:

    """
    Symbolize the two time series X and Y with a pattern window of w and a sliding time of s. WARNING:
        if len(X) - w % s != 0, X will be truncated.

        :param x: "Sequence of evenly spaced observations."
        :param w: (aka embedding dimension) Symbol window length. Leads to ensemble size of w! and
        w-tuple of symbols. e.g. w=3 => {(3,2,1), (2,1,3), ...}
        :param s: (aka time delay) Sliding window, so that a window of size w slides s discrete time points to create
        the next point in the symbolic output. Default = 1.

        :return symX: list of lists of length w where each entry is an integer rank in ascending order
    """

    lenx = len(x)
    assert not lenx < w, "Window larger than length of time series."

    symx: list[tuple[int]] = list() # accumulating symbols in this list

    idx = 0
    while not idx > (lenx-w):
        if idx == 0:
            l = x[idx:idx+w]
            sorted_list = sorted(l)
            rank_dict: dict[float, int] = {value: i for i, value in enumerate(sorted_list)}
            ranked_list = tuple(rank_dict[x] + 1 for x in l) # Adding 1 to start ranking from 1
        else:
            l = x[idx+s:idx+s+w]
            sorted_list = sorted(l)
            rank_dict: dict[float, int] = {value: i for i, value in enumerate(sorted_list)}
            ranked_list = tuple(rank_dict[x] + 1 for x in l)  # Adding 1 to start ranking from 1

        symx.append(ranked_list)
        idx += s+w

    return symx


def symbolic_transfer_entropy(x: list[float],
                              y: list[float],
                              w: int,
                              s: int = 1
                              ) -> float:

    """
    Calculates the symbolic transfer entropy of time series y driving time series x.

    :param x: random variable
    :param y: random variable
    :param w: symbol window length
    :param s: sliding window time
    :return: transfer entropy from x to y, in bits
    """

    T = len(x)

    # define the ensemble space
    ensemble = Symbol(w)

    # symbolize both time series
    symx = symbolize(x,w,s)
    symy = symbolize(y,w,s)

    # convert symbols to labels for reduced complexity
    labeled_symx = [ensemble.labels[i] for i in symx]
    labeled_symy = [ensemble.labels[i] for i in symy]

    del symx
    del symy

    labels = list(ensemble.labels.keys())

    # creating the outcome space
    pspace = np.zeros((len(labels),)*3)

    for t in range(T-1):
        pspace[labeled_symx[t+1], labeled_symx[t], labeled_symy[t]] += 1

    norm = np.sum(pspace)

    assert norm == T-1 == T-1, "Matrix sum, X, and Y not equal."

    te_terms = list()

    # Calculating Transfer Entropy
    for t in range(T-1):

        # p(x_{t+1}, x_t, y_t)
        pxtxy = pspace[labeled_symx[t+1], labeled_symx[t], labeled_symy[t]] / norm

        # p(x_{t+1} | x_t, y_t)
        pcondy = (pspace[labeled_symx[t+1], labeled_symx[t], labeled_symy[t]] /
                  np.sum(pspace[:, labeled_symx[t], labeled_symy[t]]))

        # p(x_{t+1} | x_t)
        pcondx = (pspace[labeled_symx[t+1], labeled_symx[t], labeled_symy[t]] /
                  np.sum(pspace[:, labeled_symx[t], :]))

        te_terms.append(pxtxy * log2(pcondy/pcondx))

    return sum(te_terms)












