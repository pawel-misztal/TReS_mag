from helpersmag.initData import InitData
from trainModel import main as train

initData = InitData(__file__,
                    "Tres", 
                    batch_size=24,
                    epoch_count=150, 
                    eval_every_epoch=10,
                    mhsa_add_pose_everyLayer=True,
                    normalize_before=False,
                    normalize_pos_encode=False,
                    ffn_extraDropout=False,
                    ffn_dropout=0.1,
                    mhsa_dropout=0.1, 
                    lr=2e-4*4,
                    optimizer="AdamW",
                    loss_fn="MAExDynamicMarginRankingLoss",
                    loss_weights={
                        "w_mae":1,
                        "w_dmrl":1,
                        "dmrl_alpha":2,
                        "dmrl_sort":True
                    },
                    lr_scheluder={
                        "name":"StepLR",
                        "step_size":100,
                        "gamma":0.2
                    },
                    cnn_model="effnetb4",
                    optimizerStepEvery=4,
                    freeze_cnn_for_epochs=20)
train(initData)