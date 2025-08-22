

import os
import argparse
import random
import json
import numpy as np
import torch
from args import Configs
import logging



from models import TReS, Net


print('torch version: {}'.format(torch.__version__))

class Conf:
    datapath = "/home/mrpaw/Documents/mag_databases/LIVEC_or_CLIVE/ChallengeDB_release/ChallengeDB_release"
    dataset = "clive"
    seed = 2021
    svpath = "save"
    train_patch_num = 1 #50 #ilość losowych paczy na batch
    test_patch_num = 1 #50 #ilość losowych paczy na batch
    lr = 2e-5
    weight_decay = 5e-4
    batch_size = 10
    epochs = 30#3
    vesion = 1
    patch_size = 224
    droplr = 0 #1
    gpunum = 0
    network = 'resnet50'
    nheadt = 16
    num_encoder_layerst = 2
    dim_feedforwardt = 64


def main(config,device): 
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    
    folder_path = {
        'live':     config.datapath,
        'csiq':     config.datapath,
        'tid2013':  config.datapath,
        'kadid10k': config.datapath,
        'clive':    config.datapath,
        'koniq':    config.datapath,
        'fblive':   config.datapath,
        }

    img_num = {
        'live':     list(range(0, 29)),
        'csiq':     list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013':  list(range(0, 25)),
        'clive':    list(range(0, 1162)),
        'koniq':    list(range(0, 10073)),
        'fblive':   list(range(0, 39810)),
        }
    

    print('Training and Testing on {} dataset...'.format(config.dataset))
    


    
    SavePath = config.svpath
    svPath = SavePath+ config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+'/'+'sv'
    os.makedirs(svPath, exist_ok=True)
        
    
    
     # fix the seed if needed for reproducibility
    if config.seed == 0:
        pass
    else:
        print('we are using the seed = {}'.format(config.seed))
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    total_num_images = img_num[config.dataset]
    
    
    # Randomly select 80% images for training and the rest for testing
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]
    
    
    imgsTrainPath = svPath + '/' + 'train_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'
    imgsTestPath = svPath + '/' + 'test_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'

    with open(imgsTrainPath, 'w') as json_file2:
        json.dump( train_index, json_file2)
        
    with open(imgsTestPath, 'w') as json_file2:
        json.dump( test_index, json_file2)

    solver = TReS(config,device, svPath, folder_path[config.dataset], train_index, test_index,Net)
    srcc_computed, plcc_computed = solver.train(config.seed,svPath)
    
    
    
    # logging the performance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    handler = logging.FileHandler(svPath + '/LogPerformance.log')

    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    Dataset = config.dataset
    logger.info(Dataset)
    
    PrintToLogg = 'Best PLCC: {}, SROCC: {}'.format(plcc_computed,srcc_computed)
    logger.info(PrintToLogg)
    logger.info('---------------------------')



if __name__ == '__main__':
    
    # config = Configs()
    config = Conf()
    print(config)

    # if torch.cuda.is_available():
    #         if len(config.gpunum)==1:
    #             device = torch.device("cuda", index=int(config.gpunum))
    #         else:
    #             device = torch.device("cpu")

    device = "cuda"
        
    main(config,device)
    