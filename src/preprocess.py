import pandas as pd
import numpy as np
import pickle

# helper functions


def reduce_mem_usage(df, verbose=False):
    """
    reduce memory usage by downcasting data types
    from https://www.kaggle.com/harupy/m5-baseline
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def process_calendar():
    calendarDTypes = {
        "event_name_1": "category",
        "event_name_2": "category",
        "event_type_1": "category",
        "event_type_2": "category",
        "weekday": "category",
        "wm_yr_wk": "int16",
        "wday": "int8",
        "month": "int8",
        "year": "int16",
        "snap_CA": "int8",
        "snap_TX": "int8",
        "snap_WI": "int8",
    }

    # Read csv file
    calendar = pd.read_csv("../data/calendar.csv", dtype=calendarDTypes).pipe(
        reduce_mem_usage, verbose=True
    )
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["d"] = calendar["d"].apply(lambda x: int(x.strip("d_")))
    calendar["is_weekend"] = calendar["weekday"].apply(
        lambda x: x in ["Saturday", "Sunday"]
    )

    return calendar


def process_prices():
    # Correct data types for "sell_prices.csv"
    priceDTypes = {
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": "int16",
        "sell_price": "float32",
    }

    # Read csv file
    prices = pd.read_csv("../data/sell_prices.csv", dtype=priceDTypes).pipe(
        reduce_mem_usage, verbose=True
    )

    return prices


def process_sales():

    # Define all categorical columns
    catCols = {
        "id": "category",
        "item_id": "category",
        "dept_id": "category",
        "store_id": "category",
        "cat_id": "category",
        "state_id": "category",
    }

    # Read csv file
    sales = pd.read_csv("../data/sales_train_evaluation.csv", dtype=catCols).pipe(
        reduce_mem_usage, verbose=True
    )

    date_cols_rename = {x: int(x.strip("d_")) for x in sales.columns if "d_" in x}

    sales = sales.rename(columns=date_cols_rename)
    sales_ts = sales.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=date_cols_rename.values(),
        value_name="item_sales",
        var_name="d",
    )

    return sales_ts


def process_ds():
    """
    combines sales, calendar and prices dataframe
    melt dataframe to create dataset with each row is a data
    Returns: a combined dataframe
    """
    sales_df = process_sales()

    calendar_df = process_calendar()

    prices_df = process_prices()

    sales_df = sales_df.merge(calendar_df, on="d", how="left").sort_values(["id", "d"])

    sales_df = (
        sales_df.merge(prices_df, on=["wm_yr_wk", "store_id", "item_id"], how="left")
        .drop("wm_yr_wk", axis=1)
        .assign(sell_price=lambda df: df["sell_price"])
    )

    # If the price is None the product is probably not sold on
    # the given day. Add this as a parameter.
    sales_df["no_sell_price"] = sales_df["sell_price"].isna()

    # Fill sell prices with zeros as not all models can handle
    # null values
    sales_df["sell_price"] = sales_df["sell_price"].fillna(0.0)

    sales_df = parse_snap(sales_df)

    # let's save this dataframe for future use
    sales_df.to_pickle("../data/sales_df.pkl")

    return sales_df


def parse_snap(sales):
    """Parses the SNAP values such that each store uses he SNAP value
    relevant for the store. Removes the redundant SNAP columns
    afterwards.
    Args:
        sales (pd.DataFrame): Dataframe including the sales.
    Returns:
        pd.DataFrame: Sales data with new SNAP value column.
    """

    def snap_values(state):
        return sales.iloc[np.where(sales["state_id"] == state)][f"snap_{state}"]

    sales["snap"] = (
        snap_values("CA")
        .append(snap_values("TX"), verify_integrity=True)
        .append(snap_values("WI"), verify_integrity=True)
        .sort_index()
    )

    return sales.drop(["snap_CA", "snap_TX", "snap_WI"], axis=1)
