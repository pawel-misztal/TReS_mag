import torch
from typing import Dict, List
import gc

def getEmptyModelLike(model_like:Dict[str,any]):
    empty_model = {key: torch.zeros_like(value, device='cpu') for key, value in model_like.items()}
    return empty_model

def addTwoModels(avg_weights:Dict[str,any], model_weights:Dict[str,any]):
    if(avg_weights == None):
        avg_weights = getEmptyModelLike(model_weights)


    for key in model_weights:
        avg_weights[key] += model_weights[key]

    return avg_weights

def divideModelWeights(avg_weights:Dict[str,any], divideBy:int):
    for key in avg_weights:
        avg_weights[key] = avg_weights[key] / divideBy

    return avg_weights


def main(paths:List[str], savePath:str):
    avg_weights = None

    for p in paths:
        p = "/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints/" + p
        weights = torch.load(p,map_location='cpu',weights_only=False)

        avg_weights = addTwoModels(avg_weights, weights)

        del weights
        gc.collect()
        print("merged")

    avg_weights = divideModelWeights(avg_weights, len(paths))
    torch.save(avg_weights, savePath)

if __name__ == "__main__":
    # paths = ["Tres_clive_1789422817545617379_best.pth", models lastGood_var2_v*.py
    #          "Tres_clive_7117294376353700239_best.pth",
    #          "Tres_clive_719326035856858053_best.pth",
    #          "Tres_clive_776638020563880298_best.pth",
    #          "Tres_clive_8124955111128391771_best.pth"]
    paths = ["Tres_clive_1946451168681309741_best.pth", #top 4 models
             "Tres_clive_1411240645485942859_best.pth",
             "Tres_clive_1487031744987555609_best.pth",
             "Tres_clive_1789422817545617379_best.pth",]
    outName = "lastGood_var2_mergedTop4.pth"
    resPath = "/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints/" + outName
    main(paths, resPath)
