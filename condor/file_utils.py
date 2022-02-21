import json

def loadJson(samplesjson="python/configs/samples_pfnano.json",year='2017',pfnano=True):
    samples = []
    with open(samplesjson, 'r') as f:
        json_samples = json.load(f)
        for key, value in json_samples.items():
            if value == 1:
                samples.append(key)

    if pfnano:
        fname = f"fileset/pfnanoindex_{year}.json"
    else:
        fname = f"fileset/fileset_{year}_UL_NANO.json"

    fileset = {}
    with open(fname, 'r') as f:
        files = json.load(f)
        if pfnano:
            for subdir in files[year]:
                for key, flist in files[year][subdir].items():
                    if key in samples:
                        fileset[key] = ["root://cmsxrootd.fnal.gov/" + f for f in flist]
        else:
            for key, flist in files.items():
                if key in samples:
                    fileset[key] = ["root://cmsxrootd.fnal.gov/" + f for f in flist]
    return fileset
