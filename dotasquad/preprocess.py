#!/usr/bin/env python3

from itertools import cycle, islice

import opendota
import pandas as pd
import tensorflow as tf


class Drafts:
    """
    Yield Open DOTA game data for Tensorflow

    """
    heroes = opendota.hero_dict()
    N = max(heroes)

    def __init__(self, skill='medium'):
        """
        Instantiate the dataset with a given number of
        games at a given skill level

        """
        self.games = opendota.games(skill=skill)

    def binary(self, team):
        """
        Express team's draft as a vector of 0's and 1's

        """
        return [(1 if hero in team else 0) for hero in range(self.N)]

    def __iter__(self):
        """
        Generator to yield partial drafts, reconstructed from the
        complete draft

        """
        for winner, loser in self.games:
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

                yield x, y


def train_input_fn():
    """An input function for training"""
    dataset = tf.data.Dataset.from_generator(
            lambda: Drafts(skill='medium'),
            (tf.int64, tf.int64),
            (tf.TensorShape([240]), tf.TensorShape([]))
            )

    dataset = dataset.shuffle(1000)
    return dataset.make_one_shot_iterator().get_next()
