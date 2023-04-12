"""
This script performs the following tasks:
- load_mnist: load mnist dataset into numpy array
- convert_data_to_tf_dataset: convert the mnist data to tf.data.Dataset object.
"""

import logging
import os
from pathlib import Path
import gzip
from typing import Dict, Tuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from mnist_model.utils import load_config_json

logging.basicConfig(level=logging.INFO)

# Get data path from the config_file
CONFIG_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'configs', 'config_path.json')
config_path = load_config_json(CONFIG_PATH)
DATA_PATH = config_path["DATA_PATH"]


def load_mnist(path: str, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST data from `path`. The code copy from the following link:
    Reference: https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    :param path: data source path.
    :param kind: a string to represent the data tag from source ; 'train' or 'tk10'.
    :return: A tuple of:
             - image: a numpy array of image pixels.
             - labels: a numpy array of labels.
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def convert_data_to_tf_dataset() -> Tuple[Dict[str, tf.data.Dataset], Dict[str, int]]:
    """
    Load MNIST dataset and convert to tf.data.Data object.
    :return: A tuple:
            - A data dictionary of:
                - train: a collection of x_train and y_train as tf.data.Data object.
                - test: a collection of x_test and y_test as tf.data.Data object.
            - df_info: a data dictionary contains the information of dataset
    """
    # Load mnist dataset
    train_images, train_labels = load_mnist(DATA_PATH, kind='train')
    test_images, test_labels = load_mnist(DATA_PATH, kind='t10k')
    x_train = train_images.reshape(-1, 28, 28)
    x_test = test_images.reshape(-1, 28, 28)
    y_train = train_labels
    y_test = test_labels

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # print the number of samples for each train and test
    logging.info(f"Train size {x_train.shape}")
    logging.info(f"Test size {x_test.shape}")

    data_info = {"train": {"shape": x_train.shape[1:], "num_samples": x_train.shape[0]},
                 "test":  {"shape": x_test.shape[1:], "num_samples": x_test.shape[0]},
                 "num_labels": len(set(y_train))
                 }

    # Convert data to tf.data.Data object. Combining x_train and y_train as it would be easier to shuffle
    # the data before fitting to the model.
    x_train = tf.data.Dataset.from_tensor_slices(x_train)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((x_train, y_train))

    x_test = tf.data.Dataset.from_tensor_slices(x_test)
    y_test = tf.data.Dataset.from_tensor_slices(y_test)
    test_dataset = tf.data.Dataset.zip((x_test, y_test))

    return {"train": train_dataset, "test": test_dataset}, data_info


if __name__ == "__main__":
    dataset, data_info = convert_data_to_tf_dataset()
    print(data_info)
    for val in dataset["train"].take(1).as_numpy_iterator():
        x, y = val
        print(x.shape)
        print(y)
