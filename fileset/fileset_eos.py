import os
import subprocess
import json

pfnano_tag = "v2_2"
eosbase = "root://cmseos.fnal.gov/"
eosdir = f"/store/user/lpcpfnano"
users = ["jekrupa","cmantill","pharris","drankin","dryu","yihan"]

def eos_rec_search(startdir,suffix,dirs):
    dirlook = subprocess.check_output(f"eos {eosbase} ls {startdir}", shell=True).decode('utf-8').split("\n")[:-1]
    donedirs = [[] for d in dirlook]
    for di,d in enumerate(dirlook):
        if d.endswith(suffix):
            donedirs[di].append(startdir+"/"+d)
        elif d=="log":
            continue
        else:
            # print(f"Searching {d}")
            donedirs[di] = donedirs[di] + eos_rec_search(startdir+"/"+d,suffix,dirs+donedirs[di])
    donedir = [d for da in donedirs for d in da]
    return dirs+donedir

os.system(f"mkdir -p {pfnano_tag}/")

for year in ["2016","2016APV","2017","2018"]:
    sampledict = {}
    for user in users:
        try:
            eospfnano = f"{eosdir}/{user}/{pfnano_tag}/{year}"
            samples = subprocess.check_output(f"eos {eosbase} ls {eospfnano}/*/", shell=True).decode('utf-8').split("\n")[:-1]     
            for sample in samples:
                if sample not in sampledict.keys(): sampledict[sample] = {}
                datasets = subprocess.check_output(f"eos {eosbase} ls {eospfnano}/{sample}/",shell=True).decode('utf-8').split("\n")[:-1]
                for dataset in datasets:
                    # some exceptions
                    if dataset=="JetHT" and user!="jekrupa": continue
                    if sample=="WJetsToQQ" and user!="jekrupa": continue

                    curdir = f"{eospfnano}/{sample}/{dataset}"
                    dirlog = eos_rec_search(curdir,".root",[])
                    if dataset not in sampledict[sample].keys(): 
                        sampledict[sample][dataset] = dirlog
                    else:
                        print(f"repeated {sample}/{dataset} in {user}")
                        sampledict[sample][dataset] = sampledict[sample][dataset] + dirlog
                    #print(user,sample,dataset,len(dirlog))                
        except:
            pass
    with open(f"{pfnano_tag}/{year}.json", 'w') as outfile:
        json.dump(sampledict, outfile, indent=4, sort_keys=True)
