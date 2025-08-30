from pathlib import Path
import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math
import locale


locale.setlocale(locale.LC_ALL, "pl_PL")
plt.rcParams["axes.formatter.use_locale"] = True

SAVE = True
noRefPath = Path("datas/No-Reference Image Quality Assessment.xlsx")
myMethod = Path("datas/Mojametoda.xlsx")
pibiqa = Path("datas/Progress_in_BIQA.xlsx")
pibiqa_s = Path("datas/Progress_in_BIQA_synh.xlsx")
SHEET = "Sheet1"

DATASETS = [
   "LIVE",
   "CSIQ",
   "TID2013",
   "KADID",
   "CLIVE",
   "KonIQ",
   "BID"
]

MY_MODEL = "TREX"
TRES = "TReS"

METRIC_SROCC = "SROCC"
METRIC_PLCC = "PLCC"

COMP_NAMES = [
    "HFD",
    "PQR",
    "DIIVINE",
    "BRISQUE",
    "ILNIQE",
    "BIECON",
    "MEON",
    "WaDIQaM",
    # "DBCNN",
    "TIQA",
    "MetaIQA",
    "P2P-BM",
    "HyperIQA",
    # "TReS",

    "BIQI",
    "DIIVINE",
    "BRISQUE",
    "BLIINDS-II",
    # "NRSL",
    # "NR-GLBP",
    # "FRIQUEE + DBN",
    "SFA",
    "CONTRIQUE",
    # "CNN",
    "BIECON",
    "MEON",
    "DB-CNN",
    "HyperIQA",
    # "TReS",
    "RAN4IQA",
    "CYCLEIQA",
    "HFF",
    "FOSD-IQA",
    "CVC-T",

]

COMP_NAMES = np.unique(COMP_NAMES)


data_noRef:pd.DataFrame
data_my:pd.DataFrame
data_pibiqa:pd.DataFrame
data_pibiqa_s:pd.DataFrame

datas:List[pd.DataFrame]


def loadExcels():
    global data_noRef, data_my,data_pibiqa,data_pibiqa_s, datas
    data_noRef = pd.read_excel(noRefPath,SHEET)
    data_my = pd.read_excel(myMethod,SHEET)
    data_pibiqa = pd.read_excel(pibiqa,SHEET)
    data_pibiqa_s = pd.read_excel(pibiqa_s,SHEET)
    datas = [data_noRef, data_my, data_pibiqa, data_pibiqa_s]

def searchValue(model_name:str, metric:str, dataset_name):
    col_name = dataset_name + "_" + metric
    for d in datas:
        indexer = d.get("Method") == model_name
        if not indexer.any():
          continue

        if col_name not in d.columns:
           continue

        res = d.loc[indexer, col_name]
        if(len(res) == 0):
           continue
        
        if math.isnan(res.values[0]):
           continue
        return res.values[0]
    return 0
        

def drawCompPlot(dataset:str):
  srocc:List[float] = []
  plcc:List[float] = []
  names:List[str] = []

  for m in COMP_NAMES:
     srocc.append(searchValue(m, METRIC_SROCC, dataset))
     plcc.append(searchValue(m, METRIC_PLCC, dataset))
     names.append(m)

  
  srocc.append(searchValue(TRES, METRIC_SROCC, dataset))
  plcc.append(searchValue(TRES, METRIC_PLCC, dataset))
  names.append(TRES)

  srocc.append(searchValue(MY_MODEL, METRIC_SROCC, dataset))
  plcc.append(searchValue(MY_MODEL, METRIC_PLCC, dataset))
  names.append(MY_MODEL)

  combined = np.array([srocc, plcc])
  combined = combined[combined > 0]

  min_val = np.min(combined)
  max_val = np.max(combined)
  min_val = np.max([min_val, 0.8])
  dminmax = np.abs(max_val - min_val)
  
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams["font.size"] = 24
  # fig = plt.subplots(figsize=(12,9))
  plt.figure(dataset, figsize=(14,9))

  
  barW = 0.4
  brSROCC = np.arange(len(names))
  brPLCC = np.array([x + barW for x in brSROCC])
  
  alpha = 0.65
  colors_srcc = np.repeat("lightblue",len(srocc)).tolist()
  colors_srcc[-2] = ("dodgerblue",alpha)
  colors_srcc[-1] = "dodgerblue"
  colors_plcc = np.repeat("wheat", len(plcc)).tolist()
  colors_plcc[-2] = ("darkorange",alpha)
  colors_plcc[-1] = "darkorange"

  # print(colors_srcc)
  # print(colors_plcc)

  bars_srcc = plt.bar(brSROCC, srocc, barW, edgecolor = "black", color=colors_srcc,hatch="//", label="SROCC")
  bars_plcc = plt.bar(brPLCC, plcc, barW, edgecolor = "black", color=colors_plcc,hatch="\\\\", label="PLCC")

  
  plt.hlines([srocc[-1],plcc[-1]],[np.min(brSROCC)], [np.max(brPLCC)], colors=['dodgerblue','darkorange'],linestyles=['--','--'])

  plt.legend() #loc="upper right"
  upperMargin = 0.3
  marginCoef = 0.22
  plt.ylabel("Wydajność")
  plt.xticks([r + barW/2 for r in range(len(names))], names, rotation = 90)
  plt.ylim(min_val - dminmax *marginCoef, max_val + dminmax*upperMargin)
  plt.tight_layout()

  if SAVE:
    print("saving comp ", dataset)
    p = Path("dataOut")
    if(p.exists() == False):
        p.mkdir(parents=True,exist_ok=True)
    plt.savefig(p / f"comp_{dataset}.png")
  else:
    plt.show()
  

def writeRawData():
  cols:List[str] = []
  for d in DATASETS:
    cols.append(d + "_" + METRIC_SROCC)
    cols.append(d + "_" + METRIC_PLCC)

  mets = cols
  cols = ["Metoda"] + cols
  df = pd.DataFrame(columns=cols)

  names:List[str] = []

  
  for m in COMP_NAMES:
     names.append(m)

  names.append(TRES)
  names.append(MY_MODEL)

  # print(names)

  for n in names:
    colVal = []
    for d in DATASETS:
      # for c in mets:
      v_s = searchValue(n,METRIC_SROCC,d)
      # if v_s == 0:
        #  v_s = "-"
      colVal.append(v_s)  
      
      v_p = searchValue(n,METRIC_PLCC,d)
      # if v_p == 0:
        #  v_p = "-"
      colVal.append(v_p)  
      #  print(v)
    df.loc[len(df)] = [n] + colVal

  dfr = df.round(3)
  if not SAVE:
    print(dfr)
  
  p = Path("dataOut/myComp.csv")
  if(p.parent.exists() == False):
      p.parent.mkdir(parents=True,exist_ok=True)

  if SAVE:
    print("saving df")
    dfr.to_csv(p, index=False)

def getMethodNames():
    print(data_my["Method"].values)

    methodName = "TREX"
    dataset = "CLIVE"
    metric = "SROCC"
    col = dataset + "_" + metric
    indexer = data_my.get('Method') == methodName
    # print(indexer)
    if(indexer.item() == False):
      return
    res = data_my.loc[indexer,col]
    print(res.values[0])

if __name__ == "__main__":
    loadExcels()
    # print(np.unique(COMP_NAMES))
    # # getMethodNames()
    # print(DATASETS[0])
    # print(searchValue("TReS",METRIC_SROCC,DATASETS[0]))
    # print(searchValue("TREX",METRIC_SROCC,DATASETS[0]))
    # print(searchValue("notValid",METRIC_SROCC,DATASETS[0]))

    # drawCompPlot(DATASETS[0])
    # writeRawData()
    for d in DATASETS:
      drawCompPlot(d)