from helpersmag.initData import InitData
from trainModel import main as train

initData = InitData(__file__,
                    "Tres", 
                    epoch_count=150, 
                    eval_every_epoch=10,
                    mhsa_add_pose_everyLayer=True,
                    normalize_before=True,
                    normalize_pos_encode=True,
                    ffn_extraDropout=True,
                    ffn_dropout=0.5,
                    mhsa_dropout=0.5,
                    lr_scheluder={
                        "name":"StepLR",
                        "step_size":50,
                        "gamma":0.5
                    })
train(initData)