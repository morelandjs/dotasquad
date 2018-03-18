#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from preprocess import Drafts

#def load_data(ngames=10**5):
#    """Returns the opendota dataset as (train_x, train_y), (test_x, test_y)."""
#
#    games = opendota.games(ngames=ngames)
#
#    return (train_x, train_y), (test_x, test_y)
#
#def train_input_fn(features, labels, batch_size):
#    """An input function for training"""
#    # Convert the inputs to a Dataset.
#    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
#
#    # Shuffle, repeat, and batch the examples.
#    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
#
#    # Return the dataset.
#    return dataset
#
#def train_network(ngames=10**6):
#    """
#    Train a fully-connected network on Open DOTA match data.
#    Returns the learned network weights.
#
#    """
#    X, y = opendota.matches(mmr_range='low', ngames=ngames)
#
#    weights = None
#    return weights

def main():
    data = DotaDrafts(skill='medium')
    for x, y in zip(data.x, data.y):
        print(x, y)

if __name__ == "__main__":
    main()
