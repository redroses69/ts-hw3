import pandas as pd


def prepare_long_df(data_path: str, sample_ids: list) -> tuple[pd.DataFrame, pd.DataFrame]:

    train = pd.read_csv(data_path, index_col=0)
    subset = train.loc[sample_ids]
    min_len = int(subset.notna().sum(axis=1).min())
    dates = pd.date_range(start="2025-01-01", periods=min_len, freq="W")

    long_rows = []
    for sid in sample_ids:
        vals = subset.loc[sid].dropna().values[-min_len:].astype(float)
        for t, v in zip(dates, vals):
            long_rows.append({"sensor_id": sid, "timestamp": t, "value": v})

    long_df = (
        pd.DataFrame(long_rows)
        .sort_values(["sensor_id", "timestamp"])
        .reset_index(drop=True)
    )
    return train, long_df
