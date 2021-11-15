# M5 Forecasting - accuracy

This repository consists the solution for the [M5](https://www.kaggle.com/c/m5-forecasting-accuracy) forecasting challenge. This competition was launched in March 2020. The competition aimed at predicting future sales at the product level, based on historical data.

## Table of Contents

1. Setting up Python environment
2. Installing packages
3. Training the model with cross validation
4. Inference

## 1. Setting up python environment
It is recommended that you set up a virtual environment using the following steps:
For MacOS users, you can run the following commands:
```bash
# Create a virtual env called order-allocation
pyenv virtualenv 3.7.9 m5_forecasting
# Activate the virtual env
pyenv shell m5_forecasting
# Update pip
pip install --upgrade pip
```
### 2. Installing packages
First clone the [m5 repo](https://github.com/ingka-group-digital/framtid-fulfilment/).
Install the packages in requirements.txt using
```bash
pip install -r requirements.txt
```

## 3. Training the model
If you haven't done so downlaod the data from the kaggle competition website or using the Kaggle API and place it in the data folder. To train the model with cross validation run the command:
```bash
python train_crossval.py
```
Since its training on a large dataset it will take a few hours and consume all the memory on your local machine. There is a saved model from the cross validation output in the model folder whihc you can use.

## 4. Generating forecasts
Run the `main.py` file to generate forecasts. The `main.py` uses the already trained model to do inference on a test set and outputs the rmsse for the test set as well as that of baseline model and a csv with the output forecast. You can also run the notebook, which does the same thing. The notebook has additional plots showing model performance and feature importance.
To run the `main.py` file:
```bash
python main.py
```


