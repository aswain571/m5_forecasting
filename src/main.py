import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
import joblib
import pickle


def create_submission():
    """
    laod saved model and test dataframe
    we are using test as the acutals to compare
    model to
    Args:
    returns: rmse and saves output predictions to csv file
    """

    # read test/submission dataframe
    df_test = pd.read_pickle("../data/df_test.pkl")

    # read saved model
    model = joblib.load("../model/model_latest.pkl")

    label_col = ["item_sales"]
    trainCols = df_test.columns[~df_test.columns.isin(label_col)]
    X_test = df_test[trainCols]
    y_test = df_test[label_col]

    # make prediction using previuosly trained model
    y_pred = model.predict(X_test)

    y_test_ar = np.array(y_test["item_sales"])

    print("Lightgbm model rmse is:", rmse(y_pred, y_test_ar))

    #final output dataframe
    df_test['pred'] = y_pred.tolist()
    df_test.to_csv("../submission/output_with_fc.scv")


def rmse(predictions, targets):
    """root mean square error"""
    return np.sqrt(((predictions - targets) ** 2).mean())


if "__main__" == __name__:
    create_submission()
