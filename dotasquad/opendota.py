#!/usr/bin/env python3

import json
import logging
import pickle
import requests
import time
from itertools import chain, islice
from pathlib import Path

from xdg import XDG_DATA_HOME

mmr_range = dict(
        low=(1500, 2500),
        medium=(2500, 3500),
        high=(3500, 4500),
        )


def hero_names():
    """
    Dictionary from hero id's to hero names

    """
    names = {}
    endpoint = 'https://api.opendota.com/api/heroes/'
    response = requests.get(endpoint)

    for hero in json.loads(response.text):
        names.update({hero['id']: hero['localized_name']})

    names.update({0: 'none'})

    return names


def batch_of_games(skill='medium', batch_size=10**3, match_id=None):
    """
    SQL request to pull a batch of DOTA games within a specific mmr range.
    The match_id specifies the most recent game to start pulling from.

    """
    endpoint = 'https://api.opendota.com/api/explorer?'

    radiant = 'array_agg(CASE WHEN p.player_slot < 5 THEN p.hero_id END)'
    dire = 'array_agg(CASE WHEN p.player_slot > 5 THEN p.hero_id END)'
    bracket = mmr_range[skill]

    sql_query = [
            'select m.match_id, m.radiant_win, m.avg_mmr,',
            'array_remove({}, NULL) as radiant,'.format(radiant),
            'array_remove({}, NULL) as dire'.format(dire),
            'from public_matches m',
            'join public_player_matches p on m.match_id = p.match_id',
            'where m.avg_mmr > {} and m.avg_mmr < {}'.format(*bracket),
            'and m.match_id < {}'.format(match_id) if match_id else '',
            'group by m.match_id'.format(batch_size),
            'order by m.match_id desc limit {};'.format(batch_size)
            ]

    response = requests.get(endpoint + 'sql=' + ' '.join(sql_query))

    for game in json.loads(response.text)['rows']:
        yield game


def games_gen(skill='medium'):
    """
    Generator which yields DOTA games in the specified mmr range.

    """
    logging.info("Downloading DOTA game data")
    match_id = None

    while True:
        logging.info("Retrieving games from match id: {}".format(match_id))
        time.sleep(1)

        for game in batch_of_games(skill=skill, match_id=match_id):
            keys = ('radiant', 'dire', 'radiant_win')
            radiant, dire, radiant_win = [game[k] for k in keys]

            teams = (radiant, dire) if radiant_win else (dire, radiant)
            heroes = [str(h) for h in chain.from_iterable(zip(*teams))]
            draft = ' '.join(heroes) + '\n'
            match_id = game['match_id']

            yield draft


def games(skill='medium', ngames=10**4):
    """
    Returns a list of DOTA game drafts

    """
    cache_dir = Path(XDG_DATA_HOME, 'dota')
    if not cache_dir.exists():
        cache_dir.mkdir()

    cache_file = Path(cache_dir, 'games_{}_mmr.cache'.format(skill))

    if cache_file.exists():
        with open(str(cache_file), 'rb') as f:
            return pickle.load(f)
    else:
        games = islice(games_gen(skill=skill), ngames)
        with open(str(cache_file), 'wb') as f:
            pickle.dump(list(games), f)
        return games


def main():
    """
    Scape OpenDOTA API and save game draft data to a text file.

    """
    logging.basicConfig(level=logging.INFO)
    for game in games():
        print(game)


if __name__ == "__main__":
    main()
