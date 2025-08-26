import torch
import matplotlib.pyplot as plt
import Datasets.CLIVEDataset as clive
import Datasets.LiveDataset as live
import Datasets.BIQ2021Dataset as biq
import Datasets.CISQDataset as cisq
import Datasets.Kadid10kDataset as kadid
import Datasets.Koniq10kDataset as koniq
import Datasets.TID2013Dataset as tid
import Datasets.BIDDataset as bid
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from typing import Literal

train = clive.CLIVEDataset(clive.CLIVE_PATH,True,normalize=False)
test = clive.CLIVEDataset(clive.CLIVE_PATH,False,normalize=False)



def printHist(trainDataset:Dataset, testDataset:Dataset, datasetName:str, min, max, steps, mosDmos:Literal["MOS","DMOS"] = "MOS"):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    lbs = []
    for i, (img, lbl) in enumerate(trainDataset):
        lbs.append(lbl.item())
    for i, (img, lbl) in enumerate(testDataset):
        lbs.append(lbl.item())

    print("making",datasetName)
    print(len(lbs))
    # print(lbs)
    numbins = 5
    plt.figure(figsize=(12,9))
    bins = np.linspace(min,max,steps+1)
    plt.hist(lbs,bins=bins, align="mid",edgecolor="black")
    # plt.xticks(range(min,max,step))
    plt.xlabel(f"wartości {mosDmos} obrazów")
    plt.ylabel("liczba obrazów")
    plt.title("Histogram wartości zbioru danych " + datasetName, fontsize=14*2)

    p = Path("viz/hist")
    if(p.exists() == False):
        p.mkdir(parents=True,exist_ok=True)

    plt.savefig( p / (datasetName + ".png"))
    print("saved")
# printHist(train, test, "CLIVE",0,101,20)



# train = biq.BIQ2021Dataset(biq.BIQ2021_PATH,True)
# test = biq.BIQ2021Dataset(biq.BIQ2021_PATH,False)
# printHist(train, test, "BIQ2021",0,1,20,"MOS")


train = bid.BIDDataset(bid.BID_PATH,True,loadImg=False)
test = bid.BIDDataset(bid.BID_PATH,False,loadImg=False)
printHist(train, test, "BID",0,5,20,"MOS")

train = live.LIVEDataset(live.LIVE_PATH,True,normalize=False,load_img=False)
test = live.LIVEDataset(live.LIVE_PATH,False,normalize=False,load_img=False)
printHist(train, test, "LIVE",0,101,20,"DMOS")


# train = cisq.CISQDataset(cisq.CISQ_PATH ,True)
# test = cisq.CISQDataset(cisq.CISQ_PATH,False)
# printHist(train, test, "CISQ",0,1,20,"DMOS")


# train = kadid.Kadid10kDataset(kadid.KADID10K_PATH ,True,normalize=False)
# test = kadid.Kadid10kDataset(kadid.KADID10K_PATH,False,normalize=False)
# printHist(train, test, "Kadid10k",1,5,20,"MOS")


# train = koniq.Koniq10kData(koniq.KONIQ10K_PATH ,True,normalize=False)
# test = koniq.Koniq10kData(koniq.KONIQ10K_PATH,False,normalize=False)
# printHist(train, test, "Koniq10k",1,5,20,"MOS")

# train = tid.TID2013Dataset(tid.TID2013_PATH ,True,normalize=False)
# test = tid.TID2013Dataset(tid.TID2013_PATH,False,normalize=False)
# printHist(train, test, "TID2013",0,9,20,"MOS")