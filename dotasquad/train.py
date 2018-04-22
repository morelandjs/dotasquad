#!/usr/bin/env python3

import argparse
import tensorflow as tf

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
        default=10**2,
        type=int,
        help='number of training steps'
        )


def main(argv):
    """
    Train a dense neural network with 6 hidden layers to predict optimal
    DOTA hero picks.

    """
    args = parser.parse_args(argv[1:])

    # Friendly and enemy hero names
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

    # Train the model.
    classifier.train(
            input_fn=lambda:opendota.train_input_fn(),
            steps=args.train_steps
            )

    # Evaluate the model.
    #prediction = classifier.predict(input_fn=lambda: d)
    #for p, exp in zip(prediction, y):
    #    class_id = p['class_ids'][0]
    #    probability = p['probabilities'][class_id]
    #    print(class_id, probability, exp)
    #    quit()

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    #predictions = classifier.predict()

    #template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    #for pred_dict, expec in zip(predictions, expected):
    #    class_id = pred_dict['class_ids'][0]
    #    probability = pred_dict['probabilities'][class_id]

    #    print(template.format(iris_data.SPECIES[class_id],
    #                          100 * probability, expec))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
