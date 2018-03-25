#!/usr/bin/env python3

import argparse
import tensorflow as tf

import preprocess

parser = argparse.ArgumentParser()

parser.add_argument(
        '--batch_size',
        default=100,
        type=int,
        help='batch size'
        )
parser.add_argument(
        '--train_steps',
        default=100,
        type=int,
        help='number of training steps'
        )


def main(argv):
    """
    Train a dense neural network with 6 hidden layers to predict optimal
    DOTA hero picks.

    """
    args = parser.parse_args(argv[1:])

    # Feature columns describe how to use the input.
    feature_columns = []
    for n in range(240):
        col = tf.feature_column.numeric_column(key=str(n))
        feature_columns.append(col)

    # Build 6 hidden layer DNN.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # Six hidden layers.
        hidden_units=[242, 242, 200, 160, 121, 121],
        # The model must choose between 120 hero classes.
        n_classes=121
        )

    # Train the model.
    classifier.train(
            input_fn=lambda:preprocess.train_input_fn(),
            steps=args.train_steps
            )

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
