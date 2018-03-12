#!/usr/bin/env python3

import tensorflow as tf
import opendota


def train_network(layers=6, ngames=10**6):
    """
    Train a fully-connected network on Open DOTA match data.
    Returns the learned network weights.

    """
    X, y = opendota.matches(mmr_range='low', ngames=ngames)

    weights = None
    return weights

def main():
    games = opendota.games()
    for g in games:
        print(g)

if __name__ == "__main__":
    main()
