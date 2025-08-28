from pathlib import Path
import pandas as pd

noRefPath = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/viz/datas/No-Reference Image Quality Assessment.xlsx")
myMethod = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/viz/datas/Mojametoda.xlsx")
pibiqa = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/viz/datas/Progress_in_BIQA.xlsx")
pibiqa_s = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/viz/datas/Progress_in_BIQA_synh.xlsx")
SHEET = "Sheet1"

def getMethodNames():
    xls = pd.read_excel(myMethod,SHEET)
    print(xls["Method"].values)

    methodName = "TREX"
    dataset = "CLIVE"
    metric = "SROCC"
    col = dataset + "_" + metric
    res = xls.loc[xls.get('Method') == methodName,col]
    print(res.values[0])

if __name__ == "__main__":
    getMethodNames()