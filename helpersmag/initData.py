from dataclasses import dataclass, asdict,  field
import json
from typing import Dict, List
import hashlib
import base64

@dataclass
class InitData:
    start_file_path:str
    model_class_name:str
    version:int=0
    seed:int=2137
    eval_repeats:int = 50
    eval_every_epoch:int = 10
    dataset:str="clive"
    dataset_normalized:bool=False
    batch_size:int = 32
    epoch_count:int = 150
    loss_fn:str='L1Loss'
    loss_weights:Dict = field(default_factory=lambda: {}) #if loss function has some weights, it can be stored here
    save_model:bool = True
    optimizer:str = "Adam"
    lr:float = 2e-5
    weight_decay:float = 5e-4
    optimiezr_opts:Dict = field(default_factory=lambda: {}) #additional options for optimizer
    lr_scheluder:Dict = field(default_factory=lambda: {})
    train_transform:str = "default"
    test_transform:str = "default"

    freeze_cnn_for_epochs:int = 0
    cnn_model:str = "resnet50"

    mhsa_dropout:float=0.5
    ffn_dropout:float=0.5
    fc_last_dropout:float=0
    fc_trans_dropout:float=0
    ffn_extraDropout:bool=True
    extraNormalizeAfterTrans:bool=False
    ffn_size:int=64
    num_trans_encoders:int = 2
    init_xavier:bool = True
    normalize_pos_encode:bool=False
    normalize_before:bool=False
    mhsa_add_pose_everyLayer:bool=False
    one_more_linear:bool = False
    l2_pool_paper:bool = False

    optimizerStepEvery:int = 1


    def getName(self) -> str:
        return self.model_class_name  + "_" + self.dataset + "_" + str(hash(self))

    def __hash__(self):
        base = InitData(self.start_file_path, self.model_class_name)  # base object
        #creating new dict only with different values
        changed_fields = {
            k: v for k, v in self.__dict__.items()
            if getattr(base, k, None) != v
        }
        as_str = json.dumps(changed_fields, sort_keys=True)
        h = hashlib.sha256(as_str.encode()).digest()
        short_hash_bytes = h[:8] 
        return int.from_bytes(short_hash_bytes, byteorder='big')
    # def __hash__(self):
    #     as_str = json.dumps(self.__dict__, sort_keys=True)
    #     h = hashlib.sha256(as_str.encode()).digest()
    #     short_hash_bytes = h[:8] 
    #     return int.from_bytes(short_hash_bytes, byteorder='big')
        # return base64.urlsafe_b64encode(h).decode()[:16]
    

if __name__ == '__main__':
    foo = InitData("egPath","egClassName",ffn_size=65)
    foojson = json.dumps(foo.__dict__)
    foo2 = json.loads(foojson)
    print(hash(foo))
    # print(hash(foo2))


    foo3 = InitData(**foo2)
    # print(hash(foo3))
    print(foo3.seed)
    print(foo3)