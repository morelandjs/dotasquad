#!/usr/bin/env python3

import argparse
from itertools import islice
import tensorflow as tf

import numpy as np
import opendota


parser = argparse.ArgumentParser()

parser.add_argument(
        '--batch_size',
        default=100,
        type=int,
        help='batch size'
        )
parser.add_argument(
        '--train_steps',
        default=10**4,
        type=int,
        help='number of training steps'
        )


def train_input_fn(features, labels, batch_size=100):
    """
    Network input function which supplies dota game data to the network

    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


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

    # Build 6 hidden layer DNN.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[242, 242, 200, 160, 121, 121],
        n_classes=len(feature_columns)
        )

    # Load training data from the Open DOTA API.
    games = list(islice(opendota.games_gen((2500, 3500)), 10**4))
    X, y = zip(*games)

    features = dict(zip(hero_names, np.array(X).T))
    labels = np.array(y)

    # Train the model.
    classifier.train(
            input_fn=lambda: train_input_fn(features, labels),
            steps=args.train_steps
            )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
