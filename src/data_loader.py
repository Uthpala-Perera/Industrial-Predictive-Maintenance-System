import pandas as pd

def load_data(path):

    cols = [
        "engine_id", "cycle",
        "op1", "op2", "op3"
    ] + [f"sensor_{i}" for i in range(1, 22)]

    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = cols

    return df