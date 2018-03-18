#!/usr/bin/env python3

import argparse

import numpy as np
import tensorflow as tf

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


def main():
    """
    Train a dense neural network with 6 hidden layers to predict optimal
    DOTA hero picks.

    """
    args = parser.parse_args()

    # Import Open DOTA game data
    drafts = Drafts(skill='medium')

    # split into training and test sets
    (train_x, train_y), (test_x, test_y) = drafts.split(frac=.9)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

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
