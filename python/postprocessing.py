import argparse
import glob
import logging
import os
import pickle as pkl
import warnings

import pandas as pd
import pyarrow

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


def postprocess(years, channels, samples_dir, outpath):
    for year in years:
        for ch in channels:
            condor_dir = os.listdir(samples_dir + year)

            for sample in condor_dir:
                logging.info(f"Finding {sample} samples.")

                out_files = f"{samples_dir + year}/{sample}/outfiles/"
                parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files)

                except pyarrow.lib.ArrowInvalid:
                    # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # apply selection
                logging.info("---> Applying preselection")
                presel = {
                    "lep_fj_dr": "( ( lep_fj_dr<0.8) )",
                }

                for selection in presel:
                    logging.info(f"applying {selection} selection on {len(data)} events")
                    data = data.query(presel[selection])

                logging.info("---> Done with preselection")

                os.system(f"mkdir -p {outpath}/{sample}/outfiles/")
                data.to_parquet(f"{outpath}/{sample}/outfiles/{ch}.parquet")

                for ifile in pkl_files:
                    os.system(f"cp {out_files}/{ifile} {outpath}/{sample}/outfiles/{ifile}")


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    events_dict = postprocess(years, channels, args.samples_dir, args.outpath)
    with open(f"{args.outpath}/events_dict.pkl", "wb") as fp:
        pkl.dump(events_dict, fp)


if __name__ == "__main__":
    # e.g.
    # python postprocessing.py --years 2017 --channels ele,mu

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas")
    parser.add_argument("--samples_dir", dest="samples_dir", default="../eos/Jul21_", help="path to parquets", type=str)
    parser.add_argument("--outpath", dest="outpath", help="path of the output", type=str)

    args = parser.parse_args()

    main(args)
