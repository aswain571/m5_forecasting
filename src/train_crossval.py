import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
import joblib
import pickle


def train_with_cross_val():
    """
    Training with time series cross validations
    using randomized grid search to save time
    Args:
    train dataframe
    Returns:
    saved model with best fit parameters
    """

    np.random.seed(777)

    # read train dataframe

    df = pd.read_pickle("../data/df_train.pkl")
    train_start_date = "2016-03-01"
    train_end_date = "2016-03-25"
    val_start_date = "2016-03-26"
    val_end_date = "2016-04-23"

    label_col = ["item_sales"]
    trainCols = df.columns[~df.columns.isin(label_col)]
    X_train = df.loc[train_start_date:train_end_date][trainCols]
    y_train = df.loc[train_start_date:train_end_date][label_col]

    X_val = df.loc[val_start_date:val_end_date][trainCols]
    y_val = df.loc[val_start_date:val_end_date][label_col]


    # define a 3-fold time-series split for cross validation
    tscv = TimeSeriesSplit(n_splits=3)

    # parameters for tuninig

    param_dist = {
        "boosting_type": ["gbdt"],
        "objective": ["tweedie"],
        "tweedie_variance_power": [1.1],
        "n_estimators": [500],
        "metric": ["rmse"],
        "max_depth": [30, 50, 70],
        "num_leaves": [250, 500, 1000],
        "learning_rate": [0.03, 0.1],
        "feature_fraction": [0.5, 0.7],
        "bagging_fraction": [0.5, 0.7],
    }

    reg = lgb.LGBMRegressor()

    n_iter_search = 5

    random_search = RandomizedSearchCV(
        reg,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    # Train on the training portion of the CV
    # Validated on the validation/test portion of the CV
    # Early stopping using the test portion of the dataset

    random_search.fit(
        X_train,
        y_train,
        eval_metric="rmse",
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=1,
    )
    
    # save best model parameters
    joblib.dump(random_search.best_estimator_, '../momodel_latest.pkl')
    
if __name__ == "__main__":
    train_with_cross_val()
