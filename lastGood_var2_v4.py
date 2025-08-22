from helpersmag.initData import InitData
from trainModel import main as train

initData = InitData(__file__,
                    "Tres", 
                    version=4,
                    epoch_count=150, 
                    eval_every_epoch=10,
                    mhsa_add_pose_everyLayer=True,
                    normalize_before=False,
                    normalize_pos_encode=False,
                    ffn_extraDropout=False,
                    ffn_dropout=0.1,
                    mhsa_dropout=0.1)
train(initData)