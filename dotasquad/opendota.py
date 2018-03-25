#!/usr/bin/env python3

import json
import logging
import pickle
import requests
import time
from collections import Counter
from itertools import islice
from pathlib import Path

from xdg import XDG_DATA_HOME


mmr_range = dict(
        low=(1500, 2500),
        medium=(2500, 3500),
        high=(3500, 4500),
        )


def hero_dict():
    """
    Dictionary from hero id's to hero names

    """
    names = {}
    endpoint = 'https://api.opendota.com/api/heroes/'
    response = requests.get(endpoint)

    for hero in json.loads(response.text):
        names.update({hero['id']: hero['localized_name']})

    # Add generic hero tags for missing heroes
    for n in range(max(names)):
        if n not in names:
            names.update({n: 'hero_{}'.format(n)})

    return names


def batch_of_games(skill='medium', batch_size=1000, match_id=None):
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
            keys = ('radiant', 'dire', 'radiant_win', 'match_id')
            radiant, dire, radiant_win, match_id = [game[k] for k in keys]
            yield (radiant, dire) if radiant_win else (dire, radiant)


def games(skill='medium', update=False):
    """
    Returns a list of DOTA game drafts

    """
    cache_dir = Path(XDG_DATA_HOME, 'dota')
    if not cache_dir.exists():
        cache_dir.mkdir()

    cache_file = Path(cache_dir, 'games_{}_skill.p'.format(skill))

    if cache_file.exists() and update is False:
        with open(str(cache_file), 'rb') as f:
            return pickle.load(f)

    games = islice(games_gen(skill=skill), 10**3)

    with open(str(cache_file), 'wb') as f:
        pickle.dump(list(games), f)

    return games


def main():
    """
    Scape OpenDOTA API and save game draft data to a text file.

    """
    logging.basicConfig(level=logging.INFO)
    winner_picks = [h for (h, *_), _ in games()]
    loser_picks = [h for _, (h, *_) in games()]

    wins = Counter(winner_picks)
    picks = Counter(winner_picks + loser_picks)

    heroes = hero_dict()
    TINY = 1e-6

    best_picks = [
            (wins[n] / picks[n], hero)
            for n, hero in heroes.items()
            if picks[n] > 100
            ]

    for rank, (pwin, hero) in enumerate(
            sorted(best_picks, reverse=True), start=1):

        print("{0:03d}".format(rank), "{0:.3f}".format(pwin), hero)

if __name__ == "__main__":
    main()
