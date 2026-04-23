import pandas as pd

def add_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    df.drop("max_cycle", axis=1, inplace=True)
    return df


def add_failure_label(df, threshold=30):
    df["failure"] = (df["RUL"] < threshold).astype(int)
    return df


def drop_unused_sensors(df):
    drop_cols = [
        "sensor_1", "sensor_5", "sensor_6",
        "sensor_10", "sensor_16", "sensor_18", "sensor_19"
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def normalize_by_engine(df):
    sensor_cols = [c for c in df.columns if "sensor" in c]

    df[sensor_cols] = df.groupby("engine_id")[sensor_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    df.fillna(0, inplace=True)
    return df