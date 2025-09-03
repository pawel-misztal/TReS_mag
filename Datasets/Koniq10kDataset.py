from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# KONIQ10K_PATH = Path("/home/mrpaw/Documents/mag_databases/KonIQ-10k")
KONIQ10K_PATH = Path("ZbioryDanych/KonIQ-10k_testowy")

class Koniq10kData(Dataset):
  """
  mos - higher is better 0-5 -> 0-1\n
  real distortions
  """
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, testSize = 0.2, normalize = True, seed=2137, mos_z=False, loadImg=True) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.dataPath = path / "koniq10k_scores_and_distributions/koniq10k_scores_and_distributions.csv"
    # self.imagesPath = path / "koniq10k_1024x768/1024x768"
    self.imagesPath = path / "koniq10k_512x384/512x384"
    self.normalize = normalize

    scores = pd.read_csv(self.dataPath)
    self.mos = scores["MOS_zscore"].values if mos_z else scores["MOS"].values
    self.images = scores["image_name"].values

    length = len(self.mos)

    self.indexes = np.arange(start=0, stop=length) 


    i_train, i_test = train_test_split(self.indexes, test_size=testSize, random_state=seed, shuffle=True)

    self.indexes = i_train if train else i_test
    
    self.transform = transform
    self.load_img = loadImg

  def __len__(self):
    return len(self.indexes)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    i = self.indexes[index]
    img_path = self.imagesPath / self.images[i]
    mos = self.mos[i]

    if self.load_img:
      # print(img_path)
      img = Image.open(img_path)

      if(self.transform != None):
        img = self.transform(img)
    else:
      img = None

    if(self.normalize):
      mos = mos / 5.0

    mos = torch.tensor(mos, dtype=torch.float32).unsqueeze(0)

    return (img, mos)