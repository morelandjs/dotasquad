#!/usr/bin/env python3

import argparse
from itertools import islice
from pathlib import Path
import tensorflow as tf

from fuzzywuzzy import process
import numpy as np
import opendota
from xdg import XDG_DATA_HOME


parser = argparse.ArgumentParser()
parser.add_argument(
        '--train',
        action="store_true",
        help='retrain the network'
        )
parser.add_argument(
        '--batch_size',
        default=100,
        type=int,
        help='batch size'
        )
parser.add_argument(
        '--train_steps',
        default=10**5,
        type=int,
        help='number of training steps'
        )


def train_input_fn(features, labels, batch_size=100):
    """
    Network input function which supplies dota game data

    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size=100):
    """
    An input function for evaluation or prediction

    """
    features=dict(features)
    inputs = features if labels is None else (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset

def train_network(classifier, mmr_range=(2500, 3500), steps=10**5):

    # Read Open DOTA game data
    games = list(islice(opendota.games_gen(mmr_range), steps))
    X, y = zip(*games)

    # Standardize the input format
    features = dict(zip(hero_names, np.array(X).T))
    labels = np.array(y)

    classifier.train(
            input_fn=lambda: train_input_fn(features, labels),
            steps=args.train_steps
            )

def predict(team_heroes, enemy_heroes):


def main(argv):
    """
    Train a dense neural network with 6 hidden layers to predict optimal
    DOTA hero picks.

    """
    args = parser.parse_args(argv[1:])

    # Friendly and enemy hero names.
    heroes = opendota.hero_dict()
    hero_names = [
            '{}_{}'.format(prefix, hero)
            for prefix in ('Ally', 'Enemy')
            for hero_id, hero in sorted(heroes.items())
            ]

    # Feature columns describe how to use the input.
    feature_columns = []
    for hero in hero_names:
        col = tf.feature_column.numeric_column(key=hero, dtype=tf.int64)
        feature_columns.append(col)

    # establish model cache directory
    model_dir = Path(XDG_DATA_HOME, 'dota')
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build 6 hidden layer DNN.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[242, 242, 200, 160, 121, 121],
        n_classes=len(feature_columns),
        model_dir=model_dir
        )

    # Train the model.
    if args.train:

        # Read Open DOTA game data
        games = list(islice(opendota.games_gen((2500, 3500)), 10**5))
        X, y = zip(*games)

        # Standardize the input format
        features = dict(zip(hero_names, np.array(X).T))
        labels = np.array(y)

        classifier.train(
                input_fn=lambda: train_input_fn(features, labels),
                steps=args.train_steps
                )

    # Predict hero picks
    features = {h: np.zeros(1, dtype=int) for h in hero_names}
    features['Enemy_Witch_Doctor'] = np.ones(1)

    predictions = classifier.predict(
            input_fn=lambda:eval_input_fn(features, None)
            )

    probabilities = next(predictions)['probabilities']
    hero_prob = dict(zip(hero_names, probabilities))

    for hero in sorted(hero_prob, key=hero_prob.get, reverse=True):
        print(hero, hero_prob[hero])


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
