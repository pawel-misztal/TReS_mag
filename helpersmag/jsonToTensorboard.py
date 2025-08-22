import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

from torch.utils.tensorboard import SummaryWriter

from typing import Dict, Literal, List
from types import SimpleNamespace
from pathlib import Path
import json
from initData import InitData
import glob
import os
import re
import traceback


def main(path:Path|str):
    path = Path(path)
    with open(path, 'r') as fr:
        stats = json.load(fr)


    initData:InitData = SimpleNamespace(**stats["init_data"])
    name = Path(initData.start_file_path).name.split(".")[0] + "_" + path.name.split(".")[0]
    print("name", name)
    swp = "/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka" + "/runs" + "/" + name

    sw = SummaryWriter(swp)
    

    test:List[float] = stats["test_loss"]
    train:List[float] = stats["train_loss"]
    # for i, v in enumerate(test):
    #     # print(v)
    #     sw.add_scalar("loss/test", v,i)

    # for i, v in enumerate(train):
    #     sw.add_scalar("loss/train", v,i )

    for i, (vTest, vTrain) in enumerate(zip(test,train)):
        sw.add_scalars("loss",{
            "test":vTest,
            "train":vTrain
        }, i)


    test:List[float] = stats["test_plcc"]
    train:List[float] = stats["train_plcc"]
    for i, v in enumerate(test):
        # print(v)
        sw.add_scalar("plcc/test", v,i)
        sw.add_scalars("plcc", {"test":v},i)
    for i, v in enumerate(train):
        # print(v)
        sw.add_scalar("plcc/train", v,i)
        sw.add_scalars("plcc", {"train":v}, i)


    test:List[float] = stats["test_srcc"]
    train:List[float] = stats["train_srcc"]
    for i, v in enumerate(test):
        # print(v)
        sw.add_scalar("srcc/test", v,i)
        sw.add_scalars("srcc", {"test":v},i)
    for i, v in enumerate(train):
        # print(v)
        sw.add_scalar("srcc/train", v,i)
        sw.add_scalars("srcc", {"train":v}, i)


    plcc:List[float] = stats["eval_plcc"]
    srcc:List[float] = stats["eval_srcc"]
    mae:List[float] = stats["eval_mae"]
    epochs:List[float] = stats["eval_epoch"]

    zip()
    for i, (e,v) in enumerate(zip(epochs,plcc)):
        # print(v)
        sw.add_scalar("plcc/eval", v,e)
        sw.add_scalars("plcc", {"eval":v}, e)
    for i, (e,v) in enumerate(zip(epochs,srcc)):
        # print(v)
        sw.add_scalar("srcc/eval", v,e)
        sw.add_scalars("srcc", {"eval":v}, e)
    for i, (e,v) in enumerate(zip(epochs,mae)):
        # print(v)
        sw.add_scalar("mae/eval", v,e)
        sw.add_scalars("mae", {"eval":v}, e)

    
    # preds = stats["eval_preds"]

    # pass

    sw.close()


if __name__ == "__main__":
    folder_path = '/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints'

    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    pattern = re.compile(r'^([^_]+)_([^_]+)_([^_]+)\.json$')

    matching_files = [f for f in json_files if pattern.match(os.path.basename(f))]

    for path in matching_files:
        print(path)
        try:
            main(path)
        except:
            traceback.print_exc()