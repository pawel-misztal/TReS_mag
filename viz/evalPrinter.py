import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import json
import numpy as np
import pandas as pd
from typing import Dict, Literal
from pathlib import Path
import argparse
from typing import Tuple, List
from scipy import stats as sstats
import traceback
import re


def getBestsModels(dataset:Literal["","clive","kadid10k"], name:str, targetseed:str, writeError:bool,checkpointsFolder:Path|str=None):
    if(checkpointsFolder == None):
        path = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints")
    else: 
        path = Path(checkpointsFolder)
        if(path.is_dir() == False):
            raise Exception("Path need to be a directory", checkpointsFolder)
        
    statsPaths = list(Path(path).glob("*.json"))

    if(dataset != None):
        statsPaths = list(filter(lambda p: dataset in str(p), statsPaths))


    df = pd.DataFrame(columns=["mae", "srcc", "plcc", "path json", "path py", "seed"])

    for p in statsPaths:
        with open(p, 'r') as fr:
            stats:Dict[str,any] = json.load(fr)

        try:
            mae = np.array(stats.get("eval_mae")).min()
            srcc = np.array(stats.get("eval_srcc")).max()
            plcc = np.array(stats.get("eval_plcc")).max()
            seed = stats["init_data"]["seed"]
            
            pathjson = str(p.name)
            pathpy:str = stats.get("init_data",{}).get("start_file_path")
            pathpy = pathpy.replace("\\","/")
            pathpy = Path(pathpy).name if pathpy else None


            if name not in str(pathpy):
                continue
            pattern = rf'.*_{str(dataset)}\.py$'

            # print(re.search(pattern, str(pathpy)))
            if(re.search(pattern, str(pathpy)) is None):
                continue

            if(targetseed not in str(seed)):
                continue

            df.loc[len(df)] = [mae,srcc,plcc,pathjson,pathpy,seed]
        except:
            if writeError == True:
                traceback.print_exc()
                print("File: ",p.name)


    pd.set_option('display.max_rows',None)
    # p = Path(f"/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/viz/res/out_{dataset}.csv")
    # if(p.parent.exists() == False):
    #     p.parent.mkdir(parents=True,exist_ok=True)
    # df.to_csv(p)
    print("----------------------------------------------")
    print(df)
    print("----------------------------------------------")
    print("len",len(df))
    print("----------------------------------------------")
    print("mae ", df["mae"].values)
    print("srcc ", df["srcc"].values)
    print("plcc ", df["plcc"].values)
    print("----------------------------------------------")
    if(len(df) == 0):
        return
    print("avg mae ", df["mae"].values.mean())
    print("avg srcc ", df["srcc"].values.mean())
    print("avg plcc ", df["plcc"].values.mean())
    print("----------------------------------------------")

def parseArgs() :
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", "-d", dest="dataset", type=str,
                      default=None,
                      help="dataset, default '' means all", choices=["","clive","kadid10k","bid","tid2013","live","zkoniq10k","cisq"])
    args.add_argument("--name", "-n", dest="name", type=str,
                      default="eval",
                      help="name of the file to filter with")
    args.add_argument("--path", "-p", dest="path", type=str,
                      default=None,
                      help="path to folder with checkpoints")
    args.add_argument("--seed", "-s", dest="seed", type=str,
                      default="2137",
                      help="seed to filter with, act as string")
    
    args.add_argument("--error", "-e", dest="error",action='store_true',
                      help=" write errors to console")
  
    parsed_args = args.parse_args()
    return parsed_args.dataset, parsed_args.name, parsed_args.seed, parsed_args.error

if __name__ == "__main__":
    """
    python3 evalPrinter.py -d cisq
    """
    dataset, name, seed, error= parseArgs()

    getBestsModels(dataset, name, seed, error)