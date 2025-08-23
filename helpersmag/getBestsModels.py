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


def moving_average(x:List, w:int):
    pad_width = w // 2
    x_padded = np.pad(x, pad_width, mode='edge')
    return np.convolve(x_padded, np.ones(w)/ w, 'valid') 

def getRefs(over:Literal["mae","srcc","plcc"] = "srcc",smoothAmount:int = None):
    paths = [
        "Tres_clive_1789422817545617379.json",
        "Tres_clive_719326035856858053.json",
        "Tres_clive_776638020563880298.json",
        "Tres_clive_8124955111128391771.json",
        "Tres_clive_7117294376353700239.json"
    ]

    refs = []
    for p in paths:
        p = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints") / p
        with open(p, 'r') as fr:
            stats:Dict[str,any] = json.load(fr)
        if(over == "srcc"):
            r = moving_average(np.array(stats.get("eval_srcc")),smoothAmount).max() if smoothAmount else np.array(stats.get("eval_srcc")).max()
        if(over == "plcc"):
            r = moving_average(np.array(stats.get("eval_plcc")),smoothAmount).max() if smoothAmount else np.array(stats.get("eval_plcc")).max()
        if(over == "mae"):
            r = moving_average(np.array(stats.get("eval_mae")),smoothAmount).min() if smoothAmount else np.array(stats.get("eval_mae")).min()
        refs.append(r)

    return np.array(refs)
        

def getBestsModels(checkpointsFolder:Path|str=None, resoults:int = -1, sort:Literal["mae","srcc","plcc"] = "srcc", dataset:Literal["","clive","kadid10k"]=None, smoothAmount:int = None):
    if(checkpointsFolder == None):
        path = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints")
    else: 
        path = Path(checkpointsFolder)
        if(path.is_dir() == False):
            raise Exception("Path need to be a directory", checkpointsFolder)
        
    statsPaths = list(Path(path).glob("*.json"))

    if(dataset != None):
        statsPaths = list(filter(lambda p: dataset in str(p), statsPaths))


    df = pd.DataFrame(columns=["mae", "srcc", "plcc","p-val","best2","best4","epoch", "path json", "path py"])

    refs = getRefs(sort, smoothAmount)

    for p in statsPaths:
        with open(p, 'r') as fr:
            stats:Dict[str,any] = json.load(fr)

        try:
            mae =  moving_average(np.array(stats.get("eval_mae")),smoothAmount).min() if smoothAmount else np.array(stats.get("eval_mae")).min()
            srcc = moving_average(np.array(stats.get("eval_srcc")),smoothAmount).max() if smoothAmount else np.array(stats.get("eval_srcc")).max()
            plcc = moving_average(np.array(stats.get("eval_plcc")),smoothAmount).max() if smoothAmount else np.array(stats.get("eval_plcc")).max()
            if(sort == "srcc"):
                epoch = stats.get("eval_epoch")[np.array(stats.get("eval_srcc")).argmax()]
                n = srcc
                best2 = np.sort(np.array(stats.get("eval_srcc")))[-2]
                best4 = np.sort(np.array(stats.get("eval_srcc")))[-4]
            if(sort == "plcc"):
                epoch = stats.get("eval_epoch")[np.array(stats.get("eval_plcc")).argmax()]
                n = plcc
            if(sort == "mae"):
                epoch = stats.get("eval_epoch")[np.array(stats.get("eval_mae")).argmin()]
                n = mae
            pathjson = str(p.name)
            pathpy:str = stats.get("init_data",{}).get("start_file_path")
            pathpy = pathpy.replace("\\","/")
            pathpy = Path(pathpy).name if pathpy else None

            _, pval = sstats.ttest_1samp(refs, n)
            pval = f"{pval:.5f}"

            df.loc[len(df)] = [mae,srcc,plcc,pval,best2,best4,epoch,pathjson,pathpy]
        except:
            traceback.print_exc()
            print("File: ",p.name)

    if(sort == "srcc"):
        df = df.sort_values("srcc",ascending=False)
    if(sort == "plcc"):
        df = df.sort_values("plcc",ascending=False)
    if(sort == "mae"):
        df = df.sort_values("mae",ascending=True)

    if(resoults > 0):
        df = df.head(resoults)

    print("sorting by: ", sort)
    print("limit: ", resoults)
    print("smooth Amount: ", smoothAmount)
    pd.set_option('display.max_rows',None)
    p = Path(f"/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/viz/res/out_{dataset}.csv")
    if(p.parent.exists() == False):
        p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p)
    print(df)

def parseArgs() :
    args = argparse.ArgumentParser()
    args.add_argument("--resoults", "-r" ,dest="resoults", type=int,
                      default=-1,
                      help="how many resoults will be shown, default is -1 -> means all")
    args.add_argument("--sort", "-s", dest="sort", type=str,
                      default="srcc",
                      help="sort by which parameter, default 'srcc'", choices=["mae","srcc","plcc"])
    args.add_argument("--dataset", "-d", dest="dataset", type=str,
                      default=None,
                      help="dataset, default '' means all", choices=["","clive","kadid10k"])
    args.add_argument("--smoothAmount", "-sa", dest="smoothAmount", type=int,
                      default=None,
                      help="if set the best results will be smoothed to minimize luck value")
    parsed_args = args.parse_args()
    return parsed_args.resoults, parsed_args.sort, parsed_args.dataset, parsed_args.smoothAmount

if __name__ == "__main__":
    """
    python3 -m  helpersmag.getBestsModels -h
    """
    resoults, sort, dataset, smoothAmount = parseArgs()

    getBestsModels(resoults=resoults, sort=sort, dataset=dataset, smoothAmount=smoothAmount)