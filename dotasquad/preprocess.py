#!/usr/bin/env python3

from itertools import cycle, islice

import opendota


class Drafts:
    """
    Preprocesses OpenDota game data for Tensorflow

    """
    hero_names = opendota.hero_names()
    N = max(hero_names)

    x = []
    y = []

    def __init__(self, skill='medium'):
        """
        Instantiate the dataset with a given number of
        games at a given skill level

        """
        self.build_training_data(skill)

    def split(self, frac=.8):
        """
        Split the x and y observation data into a training set
        and a test (validation) set

        """
        dim = set([len(self.x), len(self.y)])
        assert len(dim) == 1

        n = int(frac*dim.pop())
        return (self.x[:n], self.y[:n]), (self.x[n:], self.y[n:])

    def binary(self, team):
        """
        Express team's draft as a vector of 0's and 1's

        """
        return [(1 if hero in team else 0) for hero in range(self.N+1)]

    def build_training_data(self, skill):
        """
        Generator to yield partial drafts, reconstructed from the
        complete draft

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
