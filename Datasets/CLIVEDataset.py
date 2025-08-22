from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image
from pathlib import Path
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

CLIVE_PATH = Path("/home/mrpaw/Documents/mag_databases/LIVEC_or_CLIVE/ChallengeDB_release/ChallengeDB_release")

class CLIVEDataset(Dataset):
  """
  mos - higher is better 0-100 -> 0-1\n
  real distortions
  """
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, testSize = 0.2, normalize = True, seed:int=21) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.dataPath = path / "Data"
    self.imagesPath = path / "Images"
    self.AllImages_Path = self.dataPath / "AllImages_release.mat"
    self.AllMOS_Path = self.dataPath / "AllMOS_release.mat"
    self.AllStdDev_Path = self.dataPath / "AllStdDev_release.mat"
    self.normalize = normalize

    self.allImages =  sio.loadmat(self.AllImages_Path)
    self.AllMOS =  sio.loadmat(self.AllMOS_Path)
    self.AllStdDev =  sio.loadmat(self.AllStdDev_Path)

    length = len(self.allImages['AllImages_release'])

    #skiping first 7 indexes because those were to train participans
    tmp_indexes = np.arange(start=7, stop=length) 
    # validMos = self.AllMOS[self.indexes]

    # bins = np.quantile(validMos, np.arange(0,1,0.1))
    # digitalized = np.digitize(validMos, bins)

    i_train, i_test = train_test_split(tmp_indexes, test_size=testSize, random_state=seed, shuffle=True)
    # i_train, i_test = train_test_split(self.indexes, test_size=testSize, random_state=seed, shuffle=True, stratify=digitalized)

    self.indexes = []

    # for i in i_train if train else i_test:
    #   for _ in range(50):
    #     self.indexes.append(i)

    self.indexes = i_train if train else i_test
    
    self.transform = transform

  def __len__(self):
    return len(self.indexes)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    i = self.indexes[index]
    img_path = self.imagesPath / self.allImages["AllImages_release"][i][0][0]
    mos = self.AllMOS["AllMOS_release"][0][i]

    # print(img_path)
    img = Image.open(img_path)

    if(self.transform != None):
      img = self.transform(img)

    if(self.normalize):
      mos = mos / 100.0

    mos = torch.tensor(mos, dtype=torch.float32).unsqueeze(0)

    return (img, mos)