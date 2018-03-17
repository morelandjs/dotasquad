#!/usr/bin/env python3

from itertools import cycle, islice

import numpy as np
import tensorflow as tf

import opendota


class DotaDrafts:
    """
    Preprocesses OpenDota game data for Tensorflow

    """
    hero_names = opendota.hero_names()
    N = max(hero_names)

    x = []
    y = []

    def __init__(self, skill='medium', ngames=10**4):
        """
        Instantiate the dataset with a given number of
        games at a given skill level

        """
        self.build_training_data(skill, ngames)

    def binary(self, team):
        """
        Express team's draft as a vector of 0's and 1's
        
        """
        return [(1 if hero in team else 0) for hero in range(self.N+1)]

    def build_training_data(self, skill, ngames):
        """
        Generator to yield partial drafts, reconstructed from the complete draft

        """
        for winner, loser in opendota.games(skill=skill):
            for npicked in range(10):
                draft_order = [iter(loser), iter(winner)]
                partial_draft = islice(cycle(draft_order), npicked)

                picked = [pick.__next__() for pick in partial_draft]

                winner_picked, loser_picked = [
                        [h for h in team if h in picked]
                        for team in (winner, loser)
                        ]

                x = self.binary(winner_picked) + self.binary(loser_picked)
                y = [h for h in winner if h not in winner_picked].pop(0)

                self.x.append(x)
                self.y.append(y)

#def load_data(ngames=10**5):
#    """Returns the opendota dataset as (train_x, train_y), (test_x, test_y)."""
#
#    games = opendota.games(ngames=ngames)
#
#    return (train_x, train_y), (test_x, test_y)
#
#def train_input_fn(features, labels, batch_size):
#    """An input function for training"""
#    # Convert the inputs to a Dataset.
#    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
#
#    # Shuffle, repeat, and batch the examples.
#    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
#
#    # Return the dataset.
#    return dataset
#
#def train_network(ngames=10**6):
#    """
#    Train a fully-connected network on Open DOTA match data.
#    Returns the learned network weights.
#
#    """
#    X, y = opendota.matches(mmr_range='low', ngames=ngames)
#
#    weights = None
#    return weights

def main():
    data = DotaDrafts(skill='medium', ngames=1)
    for x, y in zip(data.x, data.y):
        print(x, y)

if __name__ == "__main__":
    main()
