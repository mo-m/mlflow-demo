# mnist-model
This repository contains the demo of mlflow for a Convolution networks to classify the fashion MNIST dataset.


## Local setup 
- Setup virtual environment   
```commandline
python -m venv venv 
source venv/bin/activate 
```
- Install all dependencies 
```commandline
make install
```

## Setup Mlflow tracking service
- Open a terminal on your local machine and activate the virtual environment that was previously set up.
- Change the directory to the `outputs` folder where the `MLflow` tracking database is stored.
- Run the command mlflow ui to launch the MLflow tracking server, which will be accessible at the default port of 5000.
```commandline
mlflow ui --backend-store-uri sqlite:///meas-energy-mlflow.db
```
Note: we can check whether the port 5000 is free or not by running the following command
```commandline
lsof -i:5000
```
- Run only the following command `python -m mnist_model.training --option search` or `make search` to track hyperparameter search.





