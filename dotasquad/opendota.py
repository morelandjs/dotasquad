#!/usr/bin/env python3

from itertools import cycle, islice
import json
import requests

import numpy as np
import tensorflow as tf


def hero_dict():
    """
    Dictionary from hero id's to hero names

    """
    names = {}
    endpoint = 'https://api.opendota.com/api/heroes/'
    response = requests.get(endpoint)

    for hero in json.loads(response.text):
        hero_name = hero['localized_name'].replace('\'','').replace(' ', '_')
        hero_id = hero['id']
        names.update({hero_id: hero_name})

    for n in range(max(names)):
        if n not in names:
            names.update({n: 'Hero_{}'.format(n)})

    return names


def batch_of_games(mmr_range, batch_size=1000, match_id=None):
    """
    SQL request to pull a batch of DOTA games within a specific mmr range.
    The match_id specifies the most recent game to start pulling from.

    """
    endpoint = 'https://api.opendota.com/api/explorer?'

    radiant = 'array_agg(CASE WHEN p.player_slot < 5 THEN p.hero_id END)'
    dire = 'array_agg(CASE WHEN p.player_slot > 5 THEN p.hero_id END)'

    sql_query = [
            'select m.match_id, m.radiant_win, m.avg_mmr,',
            'array_remove({}, NULL) as radiant,'.format(radiant),
            'array_remove({}, NULL) as dire'.format(dire),
            'from public_matches m',
            'join public_player_matches p on m.match_id = p.match_id',
            'where m.avg_mmr > {} and m.avg_mmr < {}'.format(*mmr_range),
            'and m.match_id < {}'.format(match_id) if match_id else '',
            'group by m.match_id'.format(batch_size),
            'order by m.match_id desc limit {};'.format(batch_size)
            ]

    response = requests.get(endpoint + 'sql=' + ' '.join(sql_query))

    for game in json.loads(response.text)['rows']:
        yield game


def games_gen(mmr_range):
    """
    Generator which yields DOTA games in the specified mmr range.

    """
    heroes = hero_dict()
    match_id = None

    def binary(team):
        """
        Express team's draft as a vector of 0's and 1's

        """
        return [(1 if hero in team else 0) for hero in range(len(heroes))]

    while True:
        for game in batch_of_games(mmr_range, match_id=match_id):
            keys = ('radiant', 'dire', 'radiant_win', 'match_id')
            radiant, dire, radiant_win, match_id = [game[k] for k in keys]
            winner, loser = (radiant, dire) if radiant_win else (dire, radiant)

            for npicked in range(10):
                draft_order = [iter(loser), iter(winner)]
                partial_draft = islice(cycle(draft_order), npicked)

                picked = [pick.__next__() for pick in partial_draft]
                winner_picked, loser_picked = [
                        [h for h in team if h in picked]
                        for team in (winner, loser)
                        ]

                try:
                    x = binary(winner_picked) + binary(loser_picked)
                    y = [h for h in winner if h not in winner_picked].pop(0)
                    yield x, y
                except IndexError:
                    continue


#def train_input_fn(batch_size=100):
#    """
#    Network input function which supplies dota game data to the network
#
#    """
#    heroes = hero_dict()
#
#    hero_names = [
#            '{}_{}'.format(prefix, hero)
#            for prefix in ('Ally', 'Enemy')
#            for hero_id, hero in sorted(heroes.items())
#            ]
#
#    games = list(islice(games_gen((2500, 3500)), 10**5))
#    X, y = zip(*games)
#
#    dataset = tf.data.Dataset.from_tensor_slices(
#            (dict(zip(hero_names, np.array(X).T)), np.array(y))
#            )
#
#    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
#
#    return dataset


def main():
    """
    Scape OpenDOTA API and save game draft data to a text file.

    """
    for game in islice(games_gen((2500, 3500)), 10):
        print(game)

if __name__ == "__main__":
    main()
