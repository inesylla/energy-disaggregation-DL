import unittest
from datetime import datetime
import pandas as pd

import dataset


def to_dt(x):
    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")


class TestLoader(unittest.TestCase):

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

    small gaps = ts < 20 seconds
    large gaps = ts > 20 seconds
    """

    def is_equal(self, df, index, values):
        return self.assertTrue(
            (df.index == index).all() and (df.values == values).all()
        )

    def setup_ts(self, name, start, end, gaps, freq="1S"):
        index = pd.date_range(start, end, freq=freq)
        values = range(len(index))
        df = pd.DataFrame({name: values}, index)
        for gap_start, gap_end in gaps:
            mask = (df.index >= to_dt(gap_start)) & (df.index < to_dt(gap_end))

            df = df[~mask]
        return df

    def setup_scenario(self, start, end, gaps={"mains": [], "appliance1": []}):
        freq = {"mains": "1S", "appliance1": "3S"}

        mains = self.setup_ts(
            "mains", to_dt(start), to_dt(end), gaps["mains"], freq=freq["mains"]
        )
        appliance1 = self.setup_ts(
            "appliance1",
            to_dt(start),
            to_dt(end),
            gaps["appliance1"],
            freq=freq["appliance1"],
        )
        return (mains, appliance1)

    def test_aligned_nomissing(self):
        """
        Scenario:
            mains:
                coverage: 100%
                sampling period: 1 sec
            appliances:
                number of appliances: 1
                coverage: 100%
                sampling period: 3sec
            alignment:
                both series are aligned
        """
        start = "2020-01-01 00:00:00"
        end = "2020-01-01 00:00:09"

        mains, appliance1 = self.setup_scenario(start, end)
        df = dataset.NilmDataset.align(mains, appliance1)
        expected_index = appliance1.index
        expected_values = [[0, 0], [3, 1], [6, 2], [9, 3]]
        self.is_equal(df, expected_index, expected_values)

    def test_aligned_nomissing_bfill(self):
        """
        Scenario:
            mains:
                coverage: 100%
                sampling period: 1 sec
            appliances:
                number of appliances: 1
                coverage: 100%
                sampling period: 3sec
            alignment:
                both series are aligned
        """
        start = "2020-01-01 00:00:00"
        end = "2020-01-01 00:00:09"

        mains, appliance1 = self.setup_scenario(start, end)
        df = dataset.NilmDataset.align(mains, appliance1, bfill=True)

        expected_index = mains.index
        expected_values = [
            [0, 0],
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 2],
            [5, 2],
            [6, 2],
            [7, 3],
            [8, 3],
            [9, 3],
        ]
        self.is_equal(df, expected_index, expected_values)

    def test_mains_small_missing(self):
        """
        Scenario:
            mains:
                coverage: 2 x small gap in sequence
                sampling period: 1 sec
            appliances:
                number of appliances: 1
                coverage: 100%
                sampling period: 3sec
            alignment:
                both series are aligned
        """

        start = "2020-01-01 00:00:00"
        end = "2020-01-01 00:00:09"
        gaps = {
            "mains": [],
            "appliance1": [("2020-01-01 00:00:03", "2020-01-01 00:00:04")],
        }
        mains, appliance1 = self.setup_scenario(start, end, gaps)
        data = dataset.NilmDataset.impute(appliance1, gapsize=3, subseqsize=1)
        self.assertEqual(len(data), 1)

        expected_index = pd.date_range(
            appliance1.index[0], appliance1.index[-1], freq="3S"
        )
        expected_values = [[0], [2], [2], [3]]
        self.is_equal(data[0], expected_index, expected_values)

    def test_mains_large_missing(self):
        """
        Scenario:
            mains:
                coverage: 2 x large gaps in sequence
                    1 x intraday gap
                    1 x interday gap
                sampling period: 1 sec
            appliances:
                coverage: 100%
                number of appliances: 1
                sampling period: 3sec
            alignment:
                both series are aligned
        """

        start = "2020-01-01 00:00:00"
        end = "2020-01-01 00:01:00"
        gaps = {
            "mains": [],
            "appliance1": [("2020-01-01 00:00:25", "2020-01-01 00:00:45")],
        }
        mains, appliance1 = self.setup_scenario(start, end, gaps)
        data = dataset.NilmDataset.impute(appliance1, gapsize=2, subseqsize=8)
        self.assertEqual(len(data), 2)
        expected_index = pd.date_range(
            to_dt("2020-01-01 00:00:00"), to_dt("2020-01-01 00:00:24"), freq="3S"
        )
        expected_values = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
        self.is_equal(data[0], expected_index, expected_values)

        expected_index = pd.date_range(
            to_dt("2020-01-01 00:00:36"), to_dt("2020-01-01 00:01:00"), freq="3S"
        )

        expected_values = [
            [15],  # It's not the perfect imputation due non-aligned 3s (bfill)
            [15],  # It's not the perfect imputation due non-aligned 3s (bfill)
            [15],
            [15],
            [16],
            [17],
            [18],
            [19],
            [20],
        ]
        self.is_equal(data[1], expected_index, expected_values)


if __name__ == "__main__":
    unittest.main()
