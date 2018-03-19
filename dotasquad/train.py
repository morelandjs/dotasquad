#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import opendota
from preprocess import Drafts

parser = argparse.ArgumentParser()

parser.add_argument(
        '--batch_size',
        default=100,
        type=int,
        help='batch size'
        )
parser.add_argument(
        '--train_steps',
        default=1000,
        type=int,
        help='number of training steps'
        )


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def main():
    """
    Train a dense neural network with 6 hidden layers to predict optimal
    DOTA hero picks.

    """
    args = parser.parse_args()

    # hero names
    heroes = opendota.hero_dict()
    hero_names = [heroes[n] for n in sorted(heroes.keys())]

    # Import Open DOTA game data
    drafts = Drafts(skill='medium')
    x = pd.DataFrame(drafts.x)
    y = pd.DataFrame(drafts.y)

    # TODO split into training and test sets
    train_x, test_x = x, x
    train_y, test_y = y, y

    # Feature columns describe how to use the input.
    my_feature_columns = []

    for hero in hero_names:
        hero_label = 'Team_{}'.format(hero)
        my_feature_columns.append(
                tf.feature_column.numeric_column(key=hero_label)
                )

    for hero in hero_names:
        hero_label = 'Opp_{}'.format(hero)
        my_feature_columns.append(
                tf.feature_column.numeric_column(key=hero_label)
                )

    # Build 6 hidden layer DNN.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Six hidden layers.
        hidden_units=[240, 240, 200, 160, 120, 120],
        # The model must choose between 120 hero classes.
        n_classes=120
        )

    # Train the model.
    classifier.train(
            input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
            steps=args.train_steps
            )

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size)
        )

    # Output test accuracy
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == "__main__":
    main()
