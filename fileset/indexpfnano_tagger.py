import json
import subprocess


def get_children(parent):
    # print(f"DEBUG : Call to get_children({parent})")
    command = f"eos root://cmseos.fnal.gov ls -F {parent}"
    # print(command)
    result = subprocess.getoutput(command)  # , stdout=subprocess.PIPE)
    # print(result)
    return result.split("\n")


def get_subfolders(parent):
    subfolders = []
    for x in get_children(parent):
        if len(x) == 0:
            continue
        if x[-1] == "/":
            subfolders.append(x)
    return subfolders


folders_to_index = {
    "v2_3": [
        ##
        "/eos/uscms/store/user/fmokhtar/v2_3/2017/HWW",
    ],
}

# samples to exclude (needs / at the end)
samples_to_exclude = []
index_APV = {}

# Data path:
# .......................f1........................|...f2.....|..........f3.......|.....f4......|.f5.|....
# /store/user/lpcpfnano/dryu/v2_2/2017/SingleMu2017/SingleMuon/SingleMuon_Run2017C/211102_162942/0000/*root
#
# MC path:
# .......................f1........................|.......................f2..............................|..........f3.........|.....f4......|.f5.|....
# /store/user/lpcpfnano/jekrupa/v2_2/2017/WJetsToQQ/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToQQ_HT-800toInf/211108_171840/0000/*root

pfnano_version = "v2_3"
folders_to_index = folders_to_index[pfnano_version]

# for pyear in ["2016", "2016APV", "2017", "2018"]:
for pyear in ["2017"]:

    # if pyear != "2017": continue
    print(pyear)

    index = {}
    for f1 in folders_to_index:
        f1 = f1.rstrip("/")

        version = "v2_2"
        if "v2_3" in f1:
            version = "v2_3"

        year = f1.split("/")[-2]
        sample_short = f1.split("/")[-1]
        if year != pyear:
            continue

        sample_short = f1.split("/")[-1]
        print(f" {sample_short}")

        if year not in index:
            index[year] = {}
        if sample_short not in index[year]:
            index[year][sample_short] = {}

        f1_subfolders = get_subfolders(f"{f1}")
        for f2 in f1_subfolders:
            print(f"\t/{f2}")

            exclude = False
            for exclude_year, exclude_version, exclude_sample in samples_to_exclude:
                if f2 == exclude_sample and pyear == exclude_year and version == exclude_version:
                    print(f"   Excluding {sample_short}, {f2}, {version}, {pyear}")
                    exclude = True
            if exclude:
                continue

            subsample_long = f2.replace("/", "")  # This should be the actual dataset name
            f2_subfolders = get_subfolders(f"{f1}/{f2}")
            if len(f2_subfolders) == 0:
                root_files = [f"{f1}/{f2}/{x}".replace("//", "/") for x in get_children((f"{f1}/{f2}")) if x[-5:] == ".root"]
                if subsample_long not in index[year][sample_short]:
                    index[year][sample_short][subsample_long] = []
                index[year][sample_short][subsample_long].extend(root_files)

            for f3 in f2_subfolders:
                # print(f"\t\t/{f3}")
                subsample_short = f3.replace("/", "")
                if "ext1" in subsample_short:
                    print("   Ext1")

                subsample_short = subsample_short.replace("_ext1", "")
                # print(f"  {subsample_short}")

                if subsample_short not in index[year][sample_short]:
                    index[year][sample_short][subsample_short] = []

                f3_subfolders = get_subfolders(f"{f1}/{f2}/{f3}")
                if len(f3_subfolders) >= 2:
                    print(f"WARNING : Found multiple timestamps for {f1}/{f2}/{f3}")

                for f4 in f3_subfolders:  # Timestamp
                    f4_subfolders = get_subfolders(f"{f1}/{f2}/{f3}/{f4}")

                    for f5 in f4_subfolders:  # 0000, 0001, ...
                        f5_children = get_children((f"{f1}/{f2}/{f3}/{f4}/{f5}"))
                        root_files = [
                            f"{f1}/{f2}/{f3}/{f4}/{f5}/{x}".replace("//", "/") for x in f5_children if x[-5:] == ".root"
                        ]
                        if year == "2016" and "HIPM" in subsample_short:
                            if sample_short not in index_APV:
                                index_APV[sample_short] = {}
                            if subsample_short not in index_APV[sample_short]:
                                index_APV[sample_short][subsample_short] = []
                                index_APV[sample_short][subsample_short].extend(root_files)
                        else:
                            if subsample_short not in index[year][sample_short]:
                                index[year][sample_short][subsample_short] = []
                            index[year][sample_short][subsample_short].extend(root_files)
                        # print(index[year][sample_short].keys())

    if pyear == "2016APV":
        for sample_short in index_APV.keys():
            for subsample_short in index_APV[sample_short].keys():
                if sample_short not in index[pyear]:
                    index[pyear][sample_short] = {}
                if subsample_short not in index[pyear][sample_short]:
                    index[pyear][sample_short][subsample_short] = []
                index[pyear][sample_short][subsample_short] = index_APV[sample_short][subsample_short]

    with open(f"pfnanoindex_{pfnano_version}_{pyear}.json", "w") as f:
        json.dump(index, f, sort_keys=True, indent=2)
