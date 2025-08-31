import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from helpersmag.trainingUtils import prepareDataset
from types import SimpleNamespace
from typing import Dict, Literal
from pathlib import Path
import argparse


def plotres(srcc:List[float], preds:List[List[float]], initData:Dict[str,any]=None, index:int = None, size=(6,6)):
    if(index == None):
        index = np.array(srcc).argmax()

    if(initData):
        initData = SimpleNamespace(**initData)
    else:
        initData = SimpleNamespace()
        initData.seed = 2137
        initData.dataset = "clive"
        initData.dataset_normalized = False
    _, testDataset = prepareDataset(initData,[],[], load_img=False)


    pred = preds[index]
    lbl = [lbl.item() for _, (_, lbl) in enumerate(testDataset)]
    indexes = np.arange(np.array(lbl).min(),np.array(lbl).max() + 1)
    pred = np.array(pred)
    lbl = np.array(lbl)

    reg1 = np.poly1d(np.polyfit(lbl, pred, deg=1))(indexes)
    # reg2 = np.poly1d(np.polyfit(lbl, pred, deg=2))(indexes)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.figure(figsize=size)
    print("max x ", lbl.max())
    plt.scatter(lbl, pred)
    # plt.plot(indexes,reg1,color="k", alpha=0.5, linestyle='--', linewidth=1)
    # plt.plot(indexes,reg2,color="c", alpha=0.5, linestyle='--', linewidth=1)
    plt.xlabel("wartość docelowa")
    plt.ylabel("wartość przewidywana")
    max_val = 5
    min_val = 0
    plt.xlim(min_val, max_val) 
    plt.ylim(min(min_val,pred.min()), max(max_val, pred.max() + pred.max() * 0.1))
    plt.plot(indexes,indexes, color="b")
    # plt.show()


def showRes(path, drawResIndex:int = None):
    s = (6,6)

    with open(path, 'r') as fr:
        stats = json.load(fr)



    plcc = stats["eval_plcc"]
    srcc = stats["eval_srcc"]


    bestIndex = np.array(srcc).argmax()

    preds = stats["eval_preds"]
    initData = stats["init_data"]
    plotres(srcc,preds,initData,size=s,index=drawResIndex)
    print( f"Best  srcc:{np.array(srcc)[bestIndex]} plcc:{np.array(plcc)[bestIndex]}")
    plt.show()


def parseArgs() :
    args = argparse.ArgumentParser()
    args.add_argument("--path", "-p" ,dest="path", type=str,
                      default=None,
                      help="path to .json file")
    args.add_argument("--name", "-n" ,dest="name", type=str,
                      default=None,
                      help="name of .json file")
    args.add_argument("--drawResIndex", "-dri" ,dest="drawResIndex", type=int,
                      default=None,
                      help="eval pred draw index  -1->last [0...n-1]")
    parsed_args = args.parse_args()
    return parsed_args.path, parsed_args.name, parsed_args.drawResIndex


if __name__ == "__main__":
    name:str
    path, name, drawResIndex = parseArgs()
    

    
    if(path):
        resPath = path
        if not str(path).startswith("/home"):
            resPath = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints") / Path(path) 
    if(name):
        if(".json" not in name):
            name = name + ".json"
        resPath = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints") / name

    showRes(resPath,drawResIndex)