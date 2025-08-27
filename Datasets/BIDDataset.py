from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.io as sio

BID_PATH = Path("/home/mrpaw/Documents/mag_databases/BID/BID-20250703T072317Z-1-001/BID")

class BIDDataset(Dataset):
  """
  mos - higher is better 0-5 -> 0-1\n
  real distortions
  """
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, testSize = 0.2, normalize = False, seed=2137,loadImg=True) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.normalize = normalize
    self.imagesPath = path / "ImageDatabase"

    imdb = sio.loadmat(self.path / "imdb.mat")
    # print(imdb.keys())
    # print(len(imdb["images"][0][0]))
    # print(imdb["images"][0][0][0][0:10])
    # print(imdb["images"][0][0][1][0][0:10])
    # print(len(imdb["path"][0]))

    self.mos = imdb["images"][0][0][0]
    self.images = imdb["images"][0][0][1][0]

    self.indexes = np.arange(start=0, stop=len(self.mos)) 
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

    # print(type(img_path[0]))
    # print(img_path)
    if self.load_img:
      img = Image.open(img_path[0])

      if(self.transform != None):
        img = self.transform(img)
    else:
      img = None

    if(self.normalize):
      mos = mos / 5.0

    mos = torch.tensor(mos, dtype=torch.float32)

    return (img, mos)
  

if __name__ == "__main__":

    from torchvision.transforms import v2
    import matplotlib.pyplot as plt
    trainTransform = [
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomCrop((224,224)),
                # v2.ToTensor(),
                # v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
            ]
    trainTransform.insert(2,v2.Resize(size=None,max_size=512))
    trainTrans = v2.Compose(trainTransform)


    testTransform = [
                v2.RandomCrop((224,224)),
                # v2.ToTensor(),
                # v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
            ]
    testTransform.insert(0,v2.Resize(size=None,max_size=512))
    testTrans = v2.Compose(testTransform)

    vizdset = BIDDataset(BID_PATH, True, loadImg=True)
    dset = BIDDataset(BID_PATH, True, loadImg=True, transform= trainTrans)
    tdset = BIDDataset(BID_PATH, True, loadImg=True, transform= testTrans)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    img,mos =  vizdset[0]
    print(mos)
    axs[0].imshow(img)
    img2,mos2 =  dset[0]
    print(mos)
    # plt.figure()
    axs[1].imshow(img2)
    # plt.show()
    img3,mos3 =  tdset[0]
    print(mos)
    # plt.figure()
    axs[2].imshow(img3)
    # plt.show()
    plt.tight_layout()
    plt.show()
    # i = 0
    # for d in dset:
    #     i += 1
    #     print("mos",d[1])
    #     if(i > 10):
    #         break