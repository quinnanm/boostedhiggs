import warnings

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")


class LumiProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        output_location="./outfiles/",
    ):
        self._year = year
        self._output_location = output_location

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Returns a set holding 3 values: (run number, lumi block, even number)."""
        # dataset = events.metadata["dataset"]

        # # store as a parquet filec
        # lumilist = set(
        #     zip(
        #         events.run,
        #         events.luminosityBlock,
        #         events.event,
        #     )
        # )
        output = {}
        output["run"] = events.run
        output["luminosityBlock"] = events.luminosityBlock
        output["event"] = events.event

        import pandas as pd

        # convert arrays to pandas
        if not isinstance(output, pd.DataFrame):
            output = self.ak_to_pandas(output)

        import os

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        if not os.path.exists(self._output_location + "/parquet"):
            os.makedirs(self._output_location + "/parquet")
        self.save_dfs_parquet(fname, output)

        return None

    def postprocess(self, accumulator):
        return accumulator

    def save_dfs_parquet(self, fname, dfs_dict):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + "/parquet/" + fname + ".parquet")
