import json


def loadJson(samplesjson="samples_pfnano.json", year="2017", pfnano="v2_2", sampleslist=None):
    samples = []
    values = {}
    with open(samplesjson, "r") as f:
        json_samples = json.load(f)[year]
        for key, value in json_samples.items():
            if value != 0:
                if sampleslist is not None and isinstance(sampleslist, list):
                    if key in sampleslist:
                        samples.append(key)
                else:
                    samples.append(key)
            if key not in values:
                values[key] = value

    fname = f"fileset/pfnanoindex_{pfnano}_{year}.json"

    fileset = {}
    with open(fname, "r") as f:
        files = json.load(f)
        for subdir in files[year]:
            for key, flist in files[year][subdir].items():
                if key in samples:
                    fileset[key] = ["root://cmsxrootd.fnal.gov/" + f for f in flist]

    return fileset, values


def printPFNano(year='2017', pfnano="v2_2", samplesjson=None):
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
    python condor/file_utils.py --samples python/configs/samples_pfnano.json --year 2017
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
