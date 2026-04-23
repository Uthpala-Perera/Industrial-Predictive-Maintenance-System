def create_features(df):

    sensors = [c for c in df.columns if "sensor" in c]

    df = df.sort_values(["engine_id", "cycle"])

    for s in sensors:
        df[f"{s}_mean5"] = df.groupby("engine_id")[s].transform(lambda x: x.rolling(5).mean())
        df[f"{s}_std5"] = df.groupby("engine_id")[s].transform(lambda x: x.rolling(5).std())
        df[f"{s}_diff"] = df.groupby("engine_id")[s].diff()

    df.fillna(0, inplace=True)
    return df