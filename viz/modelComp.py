from types import SimpleNamespace
from helpersmag.initData import InitData
from typing import Dict, Literal, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

checkpointsPath = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints")

ref_paths = [
"Tres_clive_761952156509068320.json"
,"Tres_clive_96627729541098137.json"
,"Tres_clive_176758629169885257.json"
,"Tres_clive_1373553927847024260.json"
,"Tres_clive_7454055566203373941.json"
]

dml_paths = [
"Tres_clive_1062855308850142074.json"
,"Tres_clive_302403987053947868.json"
,"Tres_clive_752651412982040348.json"
,"Tres_clive_1886272289329716179.json"
,"Tres_clive_2133341204384094426.json"
]

dtl_paths = [
"Tres_clive_8452283878293170513.json"
,"Tres_clive_1559489598999830973.json"
,"Tres_clive_2652544611813020950.json"
,"Tres_clive_143559983090144652.json"
,"Tres_clive_457283313535931313.json"
]

ed_f_paths = [
"Tres_clive_1567290630424219496.json"
,"Tres_clive_5914906146965593904.json"
,"Tres_clive_2347777805880634537.json"
,"Tres_clive_8517024618672440160.json"
,"Tres_clive_968825850510628652.json"
]

effnet_paths = [
"Tres_clive_822802878082881730.json"
,"Tres_clive_1458972176227548636.json"
,"Tres_clive_3715148344981471509.json"
,"Tres_clive_736367861568295226.json"
,"Tres_clive_1244924195611695112.json"
]

mape_f_paths = [
"Tres_clive_4672924232761685090.json"
,"Tres_clive_6411414168086702813.json"
,"Tres_clive_1167528150321289661.json"
,"Tres_clive_6518646060954713511.json"
,"Tres_clive_95912772879425239.json"
]

nb_f_paths = [
"Tres_clive_6222694893954475070.json"
,"Tres_clive_6723357161279068945.json"
,"Tres_clive_308567719885243119.json"
,"Tres_clive_2050540424795009173.json"
,"Tres_clive_33091173434441537.json"
]

npe_f_paths = [
"Tres_clive_1049446313320139143.json"
,"Tres_clive_4872185955986988335.json"
,"Tres_clive_644109769722506659.json"
,"Tres_clive_172124539464525244.json"
,"Tres_clive_773311933224965949.json"
]

org_bests_paths = [
"Tres_clive_2021199772074146574.json",
"Tres_clive_1720333705396475587.json",
"Tres_clive_481067663587499458.json",
"Tres_clive_1128390872586752808.json",
"Tres_clive_1300079273730456092.json"
]

dr02_paths = [
"Tres_clive_5737971961030844966.json",
"Tres_clive_4449766589931245183.json",
"Tres_clive_816213726103625532.json",
"Tres_clive_3754084904759575907.json",
"Tres_clive_816213726103625532.json"
]

convnext_paths = [
"Tres_clive_6282373354032043455.json",
"Tres_clive_1275536922666524941.json",
"Tres_clive_1410257865337165936.json",
"Tres_clive_1682055911629200512.json",
"Tres_clive_6018207353725825372.json"
]

best2_paths = [
"Tres_clive_2195870764671500902.json",
"Tres_clive_2132439253949894371.json",
"Tres_clive_5567949617806866492.json",
"Tres_clive_7262409513868502776.json",
"Tres_clive_2204508173240278964.json"
]

best3_paths = [
"Tres_clive_760294007813512172.json",
"Tres_clive_577865441360539724.json",
"Tres_clive_1505220887643538886.json",
"Tres_clive_5395332614919345312.json",
"Tres_clive_6760688776062093924.json"
]

best4_paths = [
"Tres_clive_1723703182944393278.json",
"Tres_clive_1140551136897037627.json",
"Tres_clive_1877733773347527837.json",
"Tres_clive_827094505691010329.json",
"Tres_clive_1909099425441213929.json"
]

class DataPaths:
    def __init__(self, name:str, paths:List[str]):
        self.name = name
        self.paths = paths

def AsInitData(rawInitData) -> InitData:
    initData = SimpleNamespace(**rawInitData)
    return initData

def loadData(path:Path) -> Tuple[float, float, InitData]:
    stats:Dict[str,List]
    with open(path, 'r') as fr:
        stats = json.load(fr)
    initData = AsInitData(stats["init_data"])

    bestSrcc = np.array(stats.get("eval_srcc")).max()
    bestPlcc = np.array(stats.get("eval_plcc")).max()
    return (bestSrcc, bestPlcc, initData)

def loadDatas(dataPaths:List[str]) -> List[Tuple[float, float, InitData]]:
    o = []
    for dp in dataPaths:
        p = checkpointsPath / dp
        o.append(loadData(p))
    return o


pathsDict = [
    DataPaths("Model bazowy", ref_paths),
    DataPaths("Dynamic margin loss", dml_paths),
    DataPaths("Dynamic triplet loss", dtl_paths),
    DataPaths("Bez dodatkowego dropout'u", ed_f_paths),
    DataPaths("Effnetb4", effnet_paths),
    DataPaths("ConvNeXt", convnext_paths),
    DataPaths("Bez enkodowania pozycji co warstwę", mape_f_paths),
    DataPaths("Normalizacja przed", nb_f_paths),
    DataPaths("Normalizacja enkodowania pozycji", npe_f_paths),
]


def add_label(bars):
    for bar in bars:
        yval = bar.get_height()
        color  = 'black'
        va = "bottom"
        if(yval) < 0:
            va = "top"
        # if(yval < 0):
        #     color = "w" 
        plt.text(bar.get_x() + bar.get_width()/2,
                yval, 
                round(yval, 5),
                ha='center', 
                va=va,
                fontsize=24,
                color=color,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor="none"),
                rotation = 0)

def makeBarChart(ref_paths:List[str], pathsDict:List[DataPaths], sort = False, title = "",firstBase = True):
    refs = loadDatas(ref_paths) if ref_paths != None else None

    datas:List[List[Tuple[float, float, InitData]]] = []
    for dp in pathsDict:
        perf = loadDatas(dp.paths)
        datas.append(perf)


    means = []
    _means = []
    medians = []
    names = [pd.name for pd in pathsDict]
    ref_srccs:np.ndarray[float] = np.array([t[0] for t in refs]) if refs else None
    for d in datas:
        srccs:np.ndarray[float] = np.array([t[0] for t in d])
        menadiff = srccs.mean() - ref_srccs.mean() if refs else srccs.mean()
        means.append(menadiff)
        medianDiff = np.median(srccs) - np.median(ref_srccs) if refs else np.median(srccs)
        medians.append(medianDiff)
        srccs.sort()
        srccs = srccs[1:-1]
        _means.append(srccs.mean())

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    fig = plt.subplots(figsize=(12,9))

    barW = 0.25

    brMean = np.arange(len(names))
    brMedian = np.array([x + barW for x in brMean])

    if(sort): # sort
        sortI = np.argsort(means)
        means = np.array(means)[sortI]
        medians = np.array(medians)[sortI]
        names = np.array(names)[sortI]

    bars_mean = plt.bar(brMean, means, barW, edgecolor = "black", color='lightblue',hatch="//", label="średnia")
    bars_median = plt.bar(brMedian, medians, barW, edgecolor = "black", color="wheat",hatch="\\\\", label="mediana")

    min_val = np.min([means, medians])
    max_val = np.max([means, medians])
    dminmax = np.abs(max_val - min_val)

    add_label(bars_mean)
    add_label(bars_median)

    if(firstBase):
        ref = np.array([t[0] for t in datas[0]])
        # plt.axhline(ref.mean(),np.min(brMean),np.max(brMedian), color='lightblue', linestyle='--')
        plt.hlines([ref.mean(),np.median(ref)],[np.min(brMean)], [np.max(brMedian)], colors=['dodgerblue','darkorange'],linestyles=['--','--'])

    plt.ylabel("Różnica SROCC")
    plt.xticks([r + barW/2 for r in range(len(names))], names, rotation = 80)

    plt.legend() #loc="upper right"
    plt.title(title,pad=20)
    marginCoef = 0.22
    plt.ylim(min_val - dminmax *marginCoef, max_val + dminmax*marginCoef)
    plt.tight_layout()
    plt.show()

    print("data")
    sortI = np.argsort(_means)
    s_means = np.array(means)[sortI]
    s_names = np.array(names)[sortI]
    for i in range(len(names)):
        print(s_names[i], " - ", s_means[i])
    print("end")
        

def main():
    # makeBarChart(ref_paths, pathsDict, False)
    makeBarChart(None, [
    DataPaths("Bez zmian", ref_paths),
    DataPaths("Bez dodatkowego\n dropout'u", ed_f_paths),
    DataPaths("Bez enkodowania\n pozycji co warstwę", mape_f_paths),
    DataPaths("Bez normalizacji\n przed", nb_f_paths),
    DataPaths("Bez normalizacji\n enkodowania pozycji", npe_f_paths),
], False, "Wpływ modyfikacji transformera modelu, na względną wydajność", True)
    
    makeBarChart(None, [
    DataPaths("Oryginalna funkcja straty", ref_paths),
    DataPaths("Dynamiczna strata\n marginesu", dml_paths),
    DataPaths("Dynamiczna strata\n tripletu", dtl_paths),
], False,"Wpływ funkcji straty na względną wydajność modelu", True)
    
    makeBarChart(None, [
    DataPaths("Resnet 50", ref_paths),
    DataPaths("Effnetb4", effnet_paths),
    DataPaths("ConvNeXt", convnext_paths),
], False, "Wpływ pretrenowanej sieci konwolucyjnej na względną wydajność modelu", True)
    

    makeBarChart(None, [
    DataPaths("dropout 0.5", ref_paths),
    DataPaths("dropout 0.1", dr02_paths),
], False, "Wpływ wartości dropout w transformatorze\nna względną wydajność modelu", True)
    

    makeBarChart(None, [
    DataPaths("Bez zmian", ref_paths),
    DataPaths("Bez dodatkowego\n dropout'u", ed_f_paths),
    DataPaths("Bez enkodowania\n pozycji co warstwę", mape_f_paths),
    DataPaths("Bez normalizacji\n przed", nb_f_paths),
    DataPaths("Bez normalizacji\n enkodowania pozycji", npe_f_paths),
    DataPaths("Dynamiczna strata\n marginesu", dml_paths),
    DataPaths("Dynamiczna strata\n tripletu", dtl_paths),
    DataPaths("dropout 0.1", dr02_paths),
    DataPaths("Effnetb4", effnet_paths),
    DataPaths("ConvNeXt", convnext_paths),
    DataPaths("Org best", org_bests_paths),
    DataPaths("Org best 2", best2_paths),
    DataPaths("Org best 3", best3_paths),
    DataPaths("Org best 4", best4_paths),
], False, "Wpływ modyfikacji transformera modelu, na względną wydajność", True)



    

if(__name__ == "__main__"):
    main()