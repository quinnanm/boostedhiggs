import json

import yaml


def loadFiles(
    samples_yaml="samples_inclusive.yaml",
    config="mc",
    year="2017",
    pfnano="v2_2",
    sampleslist=None,
    splitname="pfnano_splitting.yaml",
):
    samples = []
    values = {}

    with open(splitname, "r") as f:
        try:
            splitting = yaml.safe_load(f)[pfnano]
        except KeyError:
            raise Exception(f"Unable to load splitting with pfnano {pfnano}")

    with open(samples_yaml, "r") as f:
        all_samples = yaml.safe_load(f)[config]
        if isinstance(all_samples, dict):
            print("all_samples", all_samples.keys())
            all_samples = all_samples[year]
        if not isinstance(all_samples, list):
            raise Exception(f"Samples in config {config} and year {year} are not part of a list")

        for sample in all_samples:
            if sampleslist is not None and isinstance(sampleslist, list):
                if sample in sampleslist:
                    samples.append(sample)
                else:
                    continue
            else:
                samples.append(sample)

            try:
                values[sample] = splitting[sample]
            except KeyError:
                raise Exception(f"Splitting for sample {sample} not found")

    fname = f"fileset/pfnanoindex_{pfnano}_{year}.json"
    fileset = {}
    with open(fname, "r") as f:
        files = json.load(f)
        for subdir in files[year]:
            for key, flist in files[year][subdir].items():
                if key in samples:
                    fileset[key] = ["root://cmseos.fnal.gov/" + f for f in flist]

    return fileset, values


def printPFNano(year="2017", pfnano="v2_2", samplesjson=None):
    samples = None
    if samplesjson:
        samples = []
        with open(samplesjson, "r") as f:
            json_samples = json.load(f)[year]
            for key, value in json_samples.items():
                if value != 0:
                    samples.append(key)

    if not samples:
        desired_keys = [
            "QCD",
            "WJets",
            "ZJets",
            "TTbar",
            "ST",
            "Single",
            "DYJets",
            "EGamma",
            "JetHT",
            "HToWW",
            "ZZ",
            "WW",
            "WZ",
        ]

    fname = f"fileset/pfnanoindex_{pfnano}_{year}.json"

    with open(fname, "r") as f:
        files = json.load(f)
        for subdir in files[year]:
            for key, flist in files[year][subdir].items():
                if samples:
                    if key in samples:
                        print(key)
                        # print('key',key)
                        # print('subdir', subdir)
                else:
                    for dk in desired_keys:
                        if dk in key:
                            print(f'"{key}":4,')
                    # print('subdir', subdir)


if __name__ == "__main__":

    """
    python condor/file_utils.py --samples python/configs/samples_mc.json --year 2017
    # or
    python condor/file_utils.py --year 2017
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--pfnano", dest="pfnano", default="v2_2", help="pfnano version")
    parser.add_argument("--samples", dest="samples", default=None, help="path to datafiles", type=str)
    args = parser.parse_args()

    printPFNano(args.year, args.pfnano, args.samples)
