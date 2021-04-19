import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import redd
import utils


class Building:
    """
    Building consumption handler - definition of appliances and main
    consumption.
    """

    def __init__(self, path, name, spec):
        self.path = path
        self.name = name

        self.mains = spec["mains"]
        self.appliances = spec["appliances"]

    def get_appliances(self):
        """
        Get list of appliances
        """
        return [x["id"] for x in self.appliances]

    def load_mains(self, start, end):
        """
        Load mains consumption from start to end time interval. Using
        dataset specific loader. Online data loader to prevent memory overrun.
        Do not save whole dataset in memory
        """
        return redd.load("mains", self.path, self.mains["channels"], start, end)

    def load_appliances(self, appliances=[], start=None, end=None):
        """
        Load appliance consumption from start to end time interval. Using
        dataset specific loader. Online data loader prevent memory overrun.
        Do not save whole dataset in memory
        """
        if not appliances:
            appliances = [x["id"] for x in self.appliances]

        # WARNING: Time series inner join. Ignoring non-synced
        # datapoints from loaded chanels
        return pd.concat(
            [
                redd.load(x["id"], self.path, x["channels"], start, end)
                for x in self.appliances
                if x["id"] in appliances
            ],
            axis=1,
            join="inner",
        )


class NilmDataset:
    """
    NILM dataset handler
    NOTE: This dataset handler is used when datset preprocessing required
          - Alignment
          - Imputation
       Not used in current analysis due already preprocessed available
       dataset (non-public available and obtained once project ongoing).
    """

    def __init__(self, spec, path):
        self.path = path
        spec = utils.load_yaml(spec)

        path = os.path.join(self.path, spec["path"])
        # Load all buildings in settings
        self.buildings = {
            x["name"]: Building(os.path.join(path, x["path"]), x["name"], x)
            for x in spec["buildings"]
        }

    def get_buildings(self):
        """
        Get list of buildings
        """
        return list(self.buildings.keys())

    def get_appliances(self, building):
        """
        Get list of appliances
        """
        return self.buildings[building].get_appliances()

    def load_mains(self, building, start=None, end=None):
        """
        Load mains consumption from start to endi time interval. Using
        dataset specific loader. Online data loader to prevent memory overrun.
        Do not save whole dataset in memory
        """
        return self.buildings[building].load_mains(start, end)

    def load_appliances(self, building, appliances=[], start=None, end=None):
        """
        Load appliance consumption from start to end time interval. Using
        dataset specific loader. Online data loader to prevent memory overrun.
        Do not save whole dataset in memory
        """
        return self.buildings[building].load_appliances(appliances, start, end)

    @staticmethod
    def align(df1, df2, bfill=False):
        """
        Align two timeseries with different acquisition frequency
        """
        # Time alignment required due different acq frequency
        if bfill:
            # Raw backward filling done
            newindex = df1.index
            df2_ = df2.reindex(newindex, method="bfill")
            df = pd.concat([df1, df2_], axis=1, join="inner")
        else:
            df = pd.concat([df1, df2], axis=1, join="inner")

        return df[~df.isnull().any(axis=1)]

    @staticmethod
    def impute(df, gapsize=20, subseqsize=28800):
        """
        Data preprocessing to impute small gaps and ignore larg gaps
        Ignore non 100% coverage days

        Extract from "Subtask Gated Networks for Non-Intrusive Load Monitoring"

        For REDD dataset,we preprocessed with the following procedure
        to handle missing values. First, we split the sequence so that the
        duration of missing values in subsequence is less than 20 seconds.
        Second,we filled the  missing values in each subsequence by
        thebackward filling method. Finally, we only used the subsequences
        with more than one-day duration
        """
        df = df.sort_index()

        start = df.index[0]
        end = df.index[-1]
        newindex = pd.date_range(start, end, freq="1S")

        # Appliance time series are not aligned to 3s (ie. 3,4 sec period)
        # Use 1sec reindex in order to align to 3sec timeserie
        df = df.reindex(newindex, method="bfill", limit=4)
        newindex = pd.date_range(start, end, freq="3S")
        mask = df.index.isin(newindex)
        df = df[mask]
        # WARNING
        # if there is a gap with more than limit number of consecutive NaNs,
        # it will only be partially filled.
        df = df.fillna(method="bfill", limit=gapsize)
        columns = df.columns

        df["rowindex"] = range(df.shape[0])
        df = df[~df.iloc[:, 0].isnull()]

        diffseq = df["rowindex"].diff()
        diffsec = df.index.to_series().diff().dt.total_seconds()
        # Find big gaps to split data in subsequences
        mask = diffseq > gapsize

        # List of continuous data subsequences
        its_index = diffsec[mask].index
        its_offset = diffsec[mask].values

        data = []
        if sum(mask) > 0:
            start = df.index[0]

            # Iterate over continuous data subsequences
            for idx, (it, offset) in enumerate(zip(its_index, its_offset)):
                end = it - pd.Timedelta(seconds=offset)
                subseq = df[start:end]

                # Check where subsquences in large enough. If the subsquence
                # is not large enough then ignore, otherwise consider it valid
                if subseq.shape[0] > subseqsize:
                    data.append(subseq[columns])
                start = it

            # Check where subsquences in large enough. If the subsquence
            # is not large enough then ignore, otherwise consider it valid
            end = df.index[-1]
            subseq = df[start:end]
            if subseq.shape[0] > subseqsize:
                data.append(subseq[columns])
        else:
            # One single subsequence (valid or invalid)
            data.append(df[columns])
        return data

    ## Filterout days without minimum amount of seconds
    # tmp = df.groupby("date").apply(lambda x: x.shape[0])
    # valid_dates = tmp[tmp >= subseqsize].index
    # mask = df["date" ].isin(valid_dates)
    # return  df[mask].drop(columns=["date"])

    def load(self, building, appliances=[], start=None, end=None, bfill=False):
        return self.impute(
            self.align(
                self.load_mains(building, start, end),
                self.load_appliances(building, appliances, start, end),
                bfill,
            )
        )

    def load_raw(self, building, appliances=[], start=None, end=None, bfill=False):
        return self.align(
            self.load_mains(building, start, end),
            self.load_appliances(building, appliances, start, end),
            bfill,
        )


class InMemoryDataset(Dataset):
    """
    Inmemory dataset
    WARNING: Not the best option due potential memory overrun but did not fail
       Not used in current analysis due already preprocessed available
       dataset (non-public available and obtained once project ongoing).
    """

    def __init__(
        self, spec, path, buildings, appliance, windowsize=34459, start=None, end=None
    ):
        super().__init__()

        self.buildings = buildings
        self.appliance = appliance
        self.windowsize = windowsize

        dataset = NilmDataset(spec, path)

        # Dataset is structured as multiple long size windows
        self.data = []
        # As sliding windows are used to acces data, a lookup-table
        # is created as sequential index to reference each sliding
        # window (long window + offset within long window).
        self.datamap = {}

        data_index = 0
        window_index = 0
        for building in buildings:
            for x in dataset.load(building, [appliance], start, end):
                # Calculate number of sliding windows in the long time window
                n_windows = x.shape[0] - windowsize + 1

                # Add loaded data to dataset
                self.data.append(x.reset_index())
                # Update data index iteraring over all sliding windows in
                # dataset. Each of the indexes in global map corresponds
                # to specific long time window and offset
                self.datamap.update(
                    {window_index + i: (data_index, i) for i in range(n_windows)}
                )
                data_index += 1

                window_index += n_windows
        self.total_size = window_index

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Each of the indexes in global map corresponds
        # to specific long time window and offset. Obtain
        # long time window and offset
        data_index, window_index = self.datamap[idx]

        # Obtain start end offset in the long time window
        start = window_index
        end = self.windowsize + window_index

        # Access data
        sample = self.data[data_index].loc[start:end, "mains"]
        target = self.data[data_index].loc[start:end, self.appliance]

        return (torch.tensor(sample.values), torch.tensor(target.values))


class InMemoryKoreaDataset(Dataset):
    """
    Inmemory dataset
    WARNING: Not the best option, due potential memory overrun but did not fail

    Arguments:
        windowsize: Sliding window size
        active_threshold: Active threshold used in classification
           Default value in paper 15W
        active_ratio: In order to prevent imbalance in data it's required
           to balance number of active/inactive appliance windows. In most
           of the cases the number of inactive windows is larger than
           the number of active windows. Active ratio forces the ratio
           between active/inactive windows by removing active/inactive
           windows (in most cases inactive windows) till fulfilling the ratio
        active_oversample: In order to prevent overfitting oversampling is done
        in active windows. This argument forces random oversampling
        active_oversample times available active windows
        transform_enabled: Used to enable data preprocessing transformation,
           in this case standardization
        transform: Transformation properties, in case of standardization
           mean and standard deviation
    """

    sample_mean = None
    sample_std = None
    target_mean = None
    target_std = None

    def __init__(
        self,
        path,
        buildings,
        appliance,
        windowsize=496,
        active_threshold=15.0,
        active_ratio=None,
        active_oversample=None,
        transform_enabled=False,
        transform=None,
    ):
        super().__init__()

        self.transform_enabled = transform_enabled

        self.appliance = appliance
        self.windowsize = windowsize
        self.active_threshold = active_threshold

        # Dataset is structured as multiple long size windows
        self.data = []
        # As sliding windows are used to acces data, a lookup-table
        # is created as sequential index to reference each sliding
        # window (long window + offset within long window).
        self.datamap = {}

        filenames = os.listdir(path)

        columns = ["main", self.appliance]

        # Using original long time windows as non-related time interval windows
        # in order to prevent mixing days and concatenating not continuous
        # data. Original data has gaps between dataset files
        self.data = [
            pd.read_csv(os.path.join(path, filename), usecols=columns, sep=",")
            for filename in filenames
            for building in buildings
            if filename.startswith(building)
        ]

        df = pd.concat(self.data)
        # Data transformation
        if transform_enabled:
            if transform:
                self.sample_mean = transform["sample_mean"]
                self.sample_std = transform["sample_std"]
                self.target_mean = transform["target_mean"]
                self.target_std = transform["target_std"]
            else:
                self.sample_mean = df["main"].mean()
                self.sample_std = df["main"].std()
                self.target_mean = df[appliance].mean()
                self.target_std = df[appliance].std()

        data_index = 0
        window_index = 0

        for subseq in self.data:
            n_windows = subseq.shape[0] - windowsize + 1  # +1 why?
            # Update data index iteraring over all sliding windows in
            # dataset. Each of the indexes in global map corresponds
            # to specific long time window and offset
            self.datamap.update(
                {window_index + i: (data_index, i) for i in range(n_windows)}
            )
            data_index += 1
            window_index += n_windows

        self.total_size = window_index

        if active_ratio:
            # Fix imbalance required
            map_indexes = list(self.datamap.keys())
            # Shuffle indexes in order to prevent oversampling using same
            # building or continuous windows
            random.shuffle(map_indexes)

            # Active and inactive buffers are used to manage classified
            # sliding windows and use them later to fix imbalance
            active_indexes = []
            inactive_indexes = []

            # Classify every sliding window as active or inactive using
            # active_threshold as threshold
            for i, index in enumerate(map_indexes):
                data_index, window_index = self.datamap[index]
                start = window_index
                end = self.windowsize + window_index

                # Retreive sliding window from data
                subseq = self.data[data_index].loc[start : (end - 1), self.appliance]
                if subseq.shape[0] != self.windowsize:
                    continue

                # Fill active and inactive buffers to be used later to
                # fix imbalance
                if (subseq > active_threshold).any():  # is there any active ?
                    active_indexes.append(index)
                else:
                    inactive_indexes.append(index)

                if (i % 1000) == 0:
                    print(
                        "Loading {0}: {1}/{2}".format(
                            self.appliance, i, len(map_indexes)
                        )
                    )
            if active_oversample:
                # If oversample is required increase representation
                active_indexes = active_indexes * active_oversample

            # Identify imbalance by calculating active/inactive ratio
            n_active = len(active_indexes)
            n_inactive = len(inactive_indexes)

            # Update number of active/inactive windows to fulfill required
            # ratio and fix imbalance
            n_inactive_ = int((n_active * (1.0 - active_ratio)) / active_ratio)
            n_active_ = int((n_inactive * active_ratio) / (1.0 - active_ratio))

            if n_inactive > n_inactive_:
                n_inactive = n_inactive_
            else:
                n_active = n_active_

            # Obtain valid indexes after imbalance analysis
            valid_indexes = active_indexes[:n_active] + inactive_indexes[:n_inactive]

            # Update datamap with fixed indexes in order to point to
            # proper sliding windows
            datamap = {}
            for dst_index, src_index in enumerate(valid_indexes):
                datamap[dst_index] = self.datamap[src_index]
            self.datamap = datamap
            self.total_size = len(self.datamap.keys())

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Loader asking for specific sliding window in specific index
        # Calculate long time window and offset in order to retrieve data
        # Input data is obtained from mains time serie, target data is
        # obtained from appliance timeserie and classification is
        # done over mains time serie
        data_index, window_index = self.datamap[idx]
        start = window_index
        end = self.windowsize + window_index

        # Retreive mains data as sample data
        sample = self.data[data_index].loc[start : (end - 1), "main"]
        # Retreive appliance data as target data
        target = self.data[data_index].loc[start : (end - 1), self.appliance]

        # Calculate classification
        classification = torch.zeros(target.values.shape[0])
        if self.active_threshold:
            classification = (target.values > self.active_threshold).astype(int)

        # WARNING: This is not the proper way as both train and test values
        # used. It's just a first approach
        if self.transform_enabled:
            # Standarization enabled
            sample = (sample - self.sample_mean) / self.sample_std
            target = (target - self.target_mean) / self.target_std

        return (
            torch.tensor(sample.values, dtype=torch.float32),  # Input
            torch.tensor(target.values, dtype=torch.float32),  # Target
            torch.tensor(classification, dtype=torch.float32),  # Classification
        )


if __name__ == "__main__":
    # Default dataset handler used to explore data in colab
    # not used in training or prediction

    spec = sys.argv[1]
    path = sys.argv[2]
    appliance = sys.argv[3]

    # NOTE: Raw dataset explorer
    # from datetime import datetime
    # import pytz
    # tz = pytz.timezone("US/Eastern")
    # start = datetime(2011, 4, 20, 0,0,0)
    # end = datetime(2011, 4, 22, 0,0,0)
    # start = tz.localize(start)
    # end = tz.localize(end)

    # building = "building1"
    # appliances = ["refrigerator"]
    # dataset = NilmDataset(spec, path)
    # raw_mains = dataset.load_mains(building)
    # raw_appliances = dataset.load_appliances(building, appliances)

    # raw_df = dataset.load_raw(building, appliances)
    # clean_df = dataset.load(building, appliances)

    # buildings = ["building1", "building2"]
    # my_dataset = InMemoryDataset(spec, path, buildings, "refrigerator")

    # NOTE: Korea dataset explorer
    buildings = ["redd_house1"]
    my_dataset = InMemoryKoreaDataset(path, buildings, appliance)
