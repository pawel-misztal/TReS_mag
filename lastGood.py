from helpersmag.initData import InitData
from trainModel import main as train

initData = InitData(__file__,
                    "Tres", 
                    epoch_count=150, 
                    eval_every_epoch=10,
                    mhsa_add_pose_everyLayer=False,
                    ffn_dropout=0.1,
                    mhsa_dropout=0.1,
                    weight_decay=1e-4,
                    lr_scheluder= {
                        "name":"CyclicLR",
                        "base_lr":1e-8,
                        "max_lr":1e-4,
                        "step_size_up": 2,
                        "step_size_down":14,
                        "mode":"exp_range"
                    })
train(initData)