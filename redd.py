import os
import pandas as pd

# Acquisition properties
#   Timezone: US/Eastern
#   Frequency: 1 Hz

channel_name = "channel_%d.dat"

timezone = "US/Eastern"


def load(name, path, channels, start=None, end=None):
    """
    REDD dataset parser. Parse REDD raw data from public
    available REDD dataset files

    Merge time series from multiple files and preprocess it
        - Filter out unrequired intervals
        - Remove duplicates
        - Create time  serie index
    """
    # WARNING: Time series inner join. Ignoring non-synced
    # datapoints from loaded channels
    df = pd.concat(
        [
            pd.read_csv(
                os.path.join(path, channel_name % channel),
                sep=" ",
                names=["timestamp", name],
            ).set_index("timestamp")
            for channel in channels
        ],
        axis=1,
        join="inner",
    )
    df = df.sum(axis=1)
    df.index = pd.to_datetime(df.index, unit="s", utc=True).tz_convert(timezone)
    df = df[~df.index.duplicated(keep="first")].sort_index()  # Remove duplicates

    if start and end:
        # Filter out unrequired data from timeseries
        start_ = df.index[0].to_pydatetime()
        end_ = df.index[-1].to_pydatetime()
        if start < start_:
            start = start_
        if end > end_:
            end = end_
        df = df[start:end]
    df.name = name
    return df.sort_index()
