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

def moving_average(x:List, w:int):
    pad_width = w // 2
    x_padded = np.pad(x, pad_width, mode='edge')
    return np.convolve(x_padded, np.ones(w)/ w, 'valid') 

def drawPlot(test:List, train:List, trainLbl:str, testLbl:str, title:str, size=(6,6)):
    epochCount = np.arange(len(test))
    plt.figure(figsize=size)
    plt.plot(epochCount,train, label=trainLbl, color="c", alpha=0.5)
    plt.plot(epochCount,test, label=testLbl, color="r", alpha=0.5)
    w =11
    train_avg = moving_average(train, w)
    test_avg = moving_average(test,w)
    plt.plot(epochCount,train_avg, color="c", linestyle='--', linewidth=1)
    plt.plot(epochCount,test_avg, color="r", linestyle='--', linewidth=1)
    plt.title(title)
    plt.legend()
    # plt.show()

def drawPlotSingle(data:List, label:str, title:str, size=(6,6), x:List = None):
    data:np.ndarray = np.array(data)
    bestIndex = data.argmax()
    if(title == "mae"):
        bestIndex = data.argmin()

    if(x == None):
        epochCount = np.arange(len(data))
    else:
        epochCount = np.array(x)

    plt.figure(figsize=size)
    plt.plot(epochCount,data, label=label, color="r")
    data_avg = moving_average(data,5)
    plt.plot(epochCount,data_avg, color="k", alpha=0.5, linestyle='--', linewidth=1)
    plt.scatter([epochCount[bestIndex]], [data[bestIndex]])
    plt.title(title)
    plt.legend()
    # plt.show()



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
    _, testDataset = prepareDataset(initData,[],[])


    pred = preds[index]
    lbl = [lbl.item() for _, (_, lbl) in enumerate(testDataset)]
    indexes = np.arange(np.array(lbl).min(),np.array(lbl).max() + 1)
    pred = np.array(pred)
    lbl = np.array(lbl)

    reg1 = np.poly1d(np.polyfit(lbl, pred, deg=1))(indexes)
    # reg2 = np.poly1d(np.polyfit(lbl, pred, deg=2))(indexes)

    plt.figure(figsize=size)
    plt.scatter(lbl, pred)
    plt.plot(indexes,reg1,color="k", alpha=0.5, linestyle='--', linewidth=1)
    # plt.plot(indexes,reg2,color="c", alpha=0.5, linestyle='--', linewidth=1)
    plt.xlabel("target")
    plt.ylabel("pred")
    plt.plot(indexes,indexes, color="b")
    # plt.show()


def showRes(path, drawResIndex:int = None):
    s = (6,6)

    with open(path, 'r') as fr:
        stats = json.load(fr)

    test = stats["test_loss"]
    train = stats["train_loss"]



    drawPlot(test,train, "train", "test", "loss",s)


    test = stats["test_plcc"]
    train = stats["train_plcc"]

    drawPlot(test,train, "train", "test", "plcc",s)

    test = stats["test_srcc"]
    train = stats["train_srcc"]

    drawPlot(test,train, "train", "test", "srcc",s)


    plcc = stats["eval_plcc"]
    srcc = stats["eval_srcc"]
    mae = stats["eval_mae"]
    epochs = stats["eval_epoch"]

    drawPlotSingle(plcc, "eval", "plcc",s, x=epochs)
    drawPlotSingle(srcc, "eval", "srcc",s, x=epochs)
    drawPlotSingle(mae, "eval", "mae",s, x=epochs)

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
    if(name):
        if(".json" not in name):
            name = name + ".json"
        resPath = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints") / name

    showRes(resPath,drawResIndex)