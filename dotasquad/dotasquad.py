#!/usr/bin/env python3

import argparse
from itertools import islice
from pathlib import Path

import fuzzywuzzy
import numpy as np
import tensorflow as tf
from xdg import XDG_DATA_HOME

import opendota


# Command line arguments for training the network.
parser = argparse.ArgumentParser()
parser.add_argument(
        '--batch-size',
        default=100,
        type=int,
        help='training data batch size'
        )
parser.add_argument(
        '--train-steps',
        default=10**5,
        type=int,
        help='number of training steps'
        )
parser.add_argument(
        '--mmr-range', nargs=2,
        default=[2500, 3500],
        type=int,
        help='open dota training data mmr range'
        )

args = parser.parse_args()

# Dictionary of hero names which are the 'features'
# of the classification problem.
heroes = opendota.hero_dict()
hero_names = [
        '{}_{}'.format(prefix, hero)
        for prefix in ('Ally', 'Enemy')
        for hero_id, hero in sorted(heroes.items())
        ]

feature_columns = []
for hero in hero_names:
    col = tf.feature_column.numeric_column(key=hero, dtype=tf.int64)
    feature_columns.append(col)

# Create network checkpoint cache directory in XDG_DATA_HOME.
model_dir = Path(XDG_DATA_HOME, 'dota')
model_dir.mkdir(parents=True, exist_ok=True)

# Create a six hidden layer DNN classifier.
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[242, 242, 200, 160, 121, 121],
    n_classes=len(feature_columns),
    model_dir=model_dir
    )


def input_fn(features, labels, batch_size=100):
    """
    Network input function which supplies training, validation and
    prediction data.

    """
    inputs = dict(features) if labels is None else (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if labels is not None:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


def train(mmr_range=(2500, 3500), steps=10**5):
    """
    Train a DNN classifier to recommend hero picks using historical
    DOTA draft data.

    """
    games = list(islice(opendota.games_gen(mmr_range), steps))
    X, y = zip(*games)

    features = dict(zip(hero_names, np.array(X).T))
    labels = np.array(y)

    classifier.train(
            input_fn=lambda: input_fn(features, labels),
            steps=args.train_steps
            )


def predict(ally_picks, enemy_picks):
    """
    Recommend the next hero given current team and enemy heroes

    """
    features = {h: np.zeros(1, dtype=int) for h in hero_names}

    for heroes, team in [(ally_picks, 'Ally'), (enemy_picks, 'Enemy')]:
        for fuzzy_hero in heroes:
            hero = fuzzywuzzy.process.extractOne(
                    '{}_{}'.format(team, fuzzy_hero), hero_names
                    )[0]
            features[hero] = np.ones(1, dtype=int)

    predictions = classifier.predict(
            input_fn=lambda: input_fn(features, None)
            )

    probabilities = next(predictions)['probabilities']
    hero_prob = dict(zip(hero_names, probabilities))
    return list(sorted(hero_prob.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    train(mmr_range=args.mmr_range, steps=args.train_steps)
