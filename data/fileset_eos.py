import os
import subprocess
import json

eosbase = "root://cmseos.fnal.gov/"
eosdir = "/store/user/cmantill/PFNano/"

dirlist = [
    #["2017_preUL", "2017preUL", []], # the third component of the list are samples to skip
    ["2017", "2017UL", []],
]

def eos_rec_search(startdir,suffix,skiplist,dirs):
    dirlook = subprocess.check_output("eos %s ls %s"%(eosbase,startdir), shell=True).decode('utf-8').split("\n")[:-1]
    donedirs = [[] for d in dirlook]
    di = 0
    for d in dirlook:
        if d.endswith(suffix):
            donedirs[di].append(startdir+"/"+d)
        elif any(skip in d for skip in skiplist):
            print("Skipping %s"%d)
        else:
            print("Searching %s"%d)
            donedirs[di] = donedirs[di] + eos_rec_search(startdir+"/"+d,suffix,skiplist,dirs+donedirs[di])
        di = di + 1
    donedir = [d for da in donedirs for d in da]
    return dirs+donedir

for dirs in dirlist:
    samples = subprocess.check_output("eos %s ls %s%s"%(eosbase,eosdir,dirs[0]), shell=True).decode('utf-8').split("\n")[:-1]
    jdict = {}
    for s in samples:
        if s in dirs[2]: continue
        print("\tRunning on %s"%s)
        curdir = "%s%s/%s"%(eosdir,dirs[0],s)
        dirlog = eos_rec_search(curdir,".root",dirs[2],[])
        if not dirlog:
            print("Empty sample skipped")
        else: 
            jdict[s] = [eosbase+d for d in dirlog]
    print(dirs[1],[s for s in jdict])
    with open("fileset_%s.json"%(dirs[1]), 'w') as outfile:
        json.dump(jdict, outfile, indent=4, sort_keys=True)
