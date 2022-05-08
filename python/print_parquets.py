import pyarrow.parquet as pq
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fname',dest='fname',required=True)
args = parser.parse_args()

data = pq.read_table(args.fname).to_pandas()
print(data.columns)
