"""
This script performs the following tasks:
- train_eval_pipeline: read dataset and shuffle the train dataset and put it into the batch.
- training_job: train the model on the given seet of parameters.
- objective: Objective function that will be used to minimise the loss during the training process.
- run_hyper_search: run hyperparameter search space to find the optimal set of parameters.
"""

import argparse
import ast
import logging
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List, Union
from functools import partial
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load TensorFlow
import tensorflow as tf

# Load mlflow tracking tools
import mlflow

# Load hyperopt for hyperparameter search
from hyperopt import fmin, tpe, STATUS_OK, Trials
from hyperopt import hp

# Load local modules
from mnist_model.data_loader import convert_data_to_tf_dataset
from mnist_model.model import SimpleModel
from mnist_model.utils import normalize_pixels, load_config_json

logging.basicConfig(level=logging.INFO)

# Output path to store models
OUTPUT_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'outputs')
MODEL_PATH = os.path.join(OUTPUT_PATH, "trained_model", "model")
CONFIG_PARAMS_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'configs', 'config_hparams.json')

# from terminal run mlflow ui --backend-store-uri sqlite:///meas-energy-mlflow.db
mlflow.set_tracking_uri(f"sqlite:///{OUTPUT_PATH}/meas-energy-mlflow.db")

tf.get_logger().setLevel('ERROR')

# Set the random seed for tensorflow, numpy and random for consistency.
SEED = 100
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def train_eval_pipeline(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    """
    Load train and test datasets from data loader and create a pipeline that contains:
      - normalise the data
      - fit the dataset in memory, cache it before shuffling for a better performance.
      - shuffle dataset.
      - Batch elements of the dataset after shuffling to get unique batches at each epoch.
    :param batch_size: batch size for train and test dataset.
    :return: prepared train and test dataset and data info.
    """
    # Load data from data loader
    dataset, data_info = convert_data_to_tf_dataset()
    # Normalise the train dataset
    ds_train = dataset["train"].map(normalize_pixels)
    # Normalise test dataset
    ds_test = dataset["test"].map(normalize_pixels)
    # Cache train data before shuffling for a better performance.
    ds_train = ds_train.cache()
    # Shuffle train dataset
    ds_train = ds_train.shuffle(data_info["num_labels"], seed=SEED)
    # Batch train dataset after shuffling to get unique batches at each epoch.
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Batch test dataset to get unique batches at each epoch.
    ds_test = ds_test.batch(batch_size)
    # Cache test data before shuffling for a better performance.
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test, data_info


def training_job(save_model_path: str = MODEL_PATH,
                 num_filter_layer_1: int = 32,
                 num_filter_layer_2: int = 64,
                 kernel_size_layers: tuple = (3, 3),
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 batch_size: int = 128,
                 num_units: int = 128,
                 num_epochs: int = 10) -> Tuple[SimpleModel, Dict[str, List[float]]]:
    """
    Train and eval model.
    :param save_model_path: path to save the trained model.
    :param num_filter_layer_1: number of filter for layer 1 of ConvNet.
    :param num_filter_layer_2: number of filter for layer 2 of ConvNet.
    :param: kernel_size_layers: kernel size for convent
    :param dropout_rate: drop out rate, default is 0.3.
    :param learning_rate: learning rate, default is 0.001.
    :param batch_size: batch size for train and test dataset, default is set to 128.
    :param num_units: number of units for the dense layer.
    :param num_epochs: number of epochs, default is 10.
    :return: A tuple:
             - model: A trained model.
             - history: history of the loss and accuracy for train and eval data
                        during model fitting.
    """
    ds_train, ds_test, data_info = train_eval_pipeline(batch_size)
    image_shape = data_info['train']["shape"]
    num_labels = data_info["num_labels"]

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{OUTPUT_PATH}/logs")
    # Define mlflow experiment name
    mlflow_experiment_name = f"model-training"
    # Setup mlflow experiment
    exp = mlflow.get_experiment_by_name(name=mlflow_experiment_name)
    if not exp:
        experiment_id = mlflow.create_experiment(name=mlflow_experiment_name,
                                                 artifact_location=f"{OUTPUT_PATH}/mlruns")
    else:
        experiment_id = exp.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=mlflow_experiment_name, nested=True):

        # Autolog the tensorflow model during the training
        mlflow.tensorflow.autolog(every_n_iter=1)
        mlflow.log_param("dropout_rate", params['dropout_rate'])
        mlflow.log_param("learning_rate", params['learning_rate'])
        mlflow.log_param("num_filter_layer_1", params['num_filter_layer_1'])
        mlflow.log_param("num_filter_layer_2", params['num_filter_layer_2'])
        mlflow.log_param("num_units", params['num_units'])
        model = SimpleModel(image_shape,
                            num_filter_layer_1,
                            num_filter_layer_2,
                            kernel_size_layers,
                            dropout_rate,
                            num_units,
                            num_labels)
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # set the logits to False
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        # Fit the model and get the history
        history = model.fit(
            ds_train,
            epochs=num_epochs,
            validation_data=ds_test,
            batch_size=batch_size,
            verbose=1,
            callbacks=[tensorboard_callback]
        )
    # Evaluate the model on the test dataset
    metrics = model.evaluate(ds_test)

    # Print the evaluation results
    logging.info(f"Test Loss: {metrics[0]}")
    logging.info(f"Test Accuracy: {metrics[1]}")
    # Save model
    model.save(save_model_path)
    return model, history


def objective(params: Dict[str, Union[int, float]],
              ds_train: tf.data.Dataset,
              ds_test: tf.data.Dataset,
              batch_size: int,
              kernel_size_layers: tuple,
              data_info: Dict,
              num_epochs: int) -> Dict[str, Union[str, float]]:
    """
    Objective function that will be used to minimise the loss during the training process.
    :param params: a dictionary of parameters that will be used to compute the loss.
    :param ds_train: training dataset.
    :param ds_test: testing dataset.
    :param batch_size: batch size to fit the model.
    :param: kernel_size_layers: kernel size for convent
    :param data_info: a data dictionary contains the information of dataset
    :param num_epochs: number of epoch to train the model.
    :return: A data dictionary:
            - loss:  a float value that attempting to minimise
            - status: Status of completion; ok' for successful completion, and 'fail' in cases where the function turned
                      out to be undefined.
    """
    image_shape = data_info['train']["shape"]
    num_labels = data_info["num_labels"]

    # Define mlflow experiment name
    mlflow_experiment_name = f"model-hyper-search"
    # Setup mlflow experiment
    exp = mlflow.get_experiment_by_name(name=mlflow_experiment_name)
    if not exp:
        experiment_id = mlflow.create_experiment(name=mlflow_experiment_name,
                                                 artifact_location=f"{OUTPUT_PATH}/mlruns")
    else:
        experiment_id = exp.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=mlflow_experiment_name, nested=True):
        # Autolog the tensorflow model during the training
        mlflow.tensorflow.autolog(every_n_iter=1)
        mlflow.log_param("dropout_rate", params['dropout_rate'])
        mlflow.log_param("learning_rate", params['learning_rate'])
        mlflow.log_param("num_filter_layer_1", params['num_filter_layer_1'])
        mlflow.log_param("num_filter_layer_2", params['num_filter_layer_2'])

        model = SimpleModel(image_shape=image_shape,
                            dropout_rate=params['dropout_rate'],
                            num_filter_layer_1=params['num_filter_layer_1'],
                            num_filter_layer_2=params['num_filter_layer_2'],
                            kernel_size_layers=kernel_size_layers,
                            num_units=params["num_units"],
                            num_labels=num_labels)


        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        model.fit(ds_train,
                  epochs=num_epochs,
                  validation_data=ds_test,
                  batch_size=batch_size,
                  verbose=1)

        # Get loss from eval model, the loss will be minimised by objective function
        eval_metrics = model.evaluate(ds_test)

    return {'loss': eval_metrics[0], 'status': STATUS_OK}


def run_hyper_search(max_eval: int, num_epochs: int, batch_size: int, kernel_size_layers: Tuple[int, int]):
    """
    Run hyperparameter search space to find the optimal set of parameters.
    :param max_eval: Maximum number of iteration to run the search space.
    :param num_epochs: number of epoch to train the model.
    :param batch_size: batch size for fitting the model.
    :param: kernel_size_layers: kernel size for convent.
    :return: Result of search space
    """
    ds_train, ds_test, data_info = train_eval_pipeline(batch_size)
    # Define the search space. This is only used for the purpose of the demo.
    # Only learning rate, dropout ratio and number of neurons considered as hyperparameter
    params = {'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
              'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
              'num_filter_layer_1': hp.quniform('num_filter_layer_1', 16, 128, 16),
              'num_filter_layer_2': hp.quniform('num_filter_layer_2', 16, 128, 16),
              'num_units': hp.quniform('num_units', 32, 256, 32),
              }
    # Create the objective function that wants to be minimised
    obj = partial(objective,
                  ds_train=ds_train,
                  ds_test=ds_test,
                  batch_size=batch_size,
                  kernel_size_layers=kernel_size_layers,
                  data_info=data_info,
                  num_epochs=num_epochs)

    # Minimise the objective over the space
    result = fmin(
        fn=obj,
        space=params,
        algo=tpe.suggest,
        max_evals=max_eval,
        trials=Trials(),
        verbose=True)
    return result


if __name__ == "__main__":
    params = load_config_json(CONFIG_PARAMS_PATH)
    parser = argparse.ArgumentParser(description='SimpleModel')
    parser.add_argument('--option', type=str, default='all', help="""Run either training or hyperparameter search 
                                                                     according to selected option. 
                                                                     The option can be one of the following: 
                                                                  - 'all' train the model and hyperparameter search. 
                                                                  - 'train' only train the model.
                                                                  - 'search' only hyperparameter search""")

    args = parser.parse_args()

    params["kernel_size_layers"] = ast.literal_eval(params["kernel_size_layers"])
    if args.option == 'all':
        # Run both training the model and do hyperparameter search
        # Only train a model
        training_job(dropout_rate=params["dropout_rate"],
                     num_filter_layer_1=params["num_filter_layer_1"],
                     num_filter_layer_2=params["num_filter_layer_2"],
                     kernel_size_layers=params["kernel_size_layers"],
                     learning_rate=params["learning_rate"],
                     batch_size=params["batch_size"],
                     num_units=params["num_units"],
                     num_epochs=params["num_epochs"])

        run_hyper_search(max_eval=params["max_eval"],
                         num_epochs=params["num_epochs"],
                         batch_size=params["batch_size"],
                         kernel_size_layers=params["kernel_size_layers"])

    elif args.option == 'train':
        # Only train a model
        training_job(dropout_rate=params["dropout_rate"],
                     num_filter_layer_1=params["num_filter_layer_1"],
                     num_filter_layer_2=params["num_filter_layer_2"],
                     kernel_size_layers=params["kernel_size_layers"],
                     learning_rate=params["learning_rate"],
                     batch_size=params["batch_size"],
                     num_units=params["num_units"],
                     num_epochs=params["num_epochs"])

    elif args.option == 'search':
        # Only perform hyperparameter search
        run_hyper_search(max_eval=params["max_eval"],
                         num_epochs=params["num_epochs"],
                         batch_size=params["batch_size"],
                         kernel_size_layers=params["kernel_size_layers"])



