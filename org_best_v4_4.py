from helpersmag.initData import InitData
from trainModel import main as train

initData = InitData(__file__,
                    "Tres", 
                    seed=21374,
                    batch_size=14,
                    epoch_count=300, 
                    eval_every_epoch=10,
                    mhsa_add_pose_everyLayer=True,
                    normalize_before=True,
                    normalize_pos_encode=True,
                    ffn_extraDropout=True,
                    ffn_dropout=0.5,
                    mhsa_dropout=0.5,
                    lr_scheluder={
                        "name":"StepLR",
                        "step_size":100,
                        "gamma":0.1
                    },
                    loss_fn="MAExDynamicMarginRankingLoss",
                    loss_weights={
                        "w_mae":1,
                        "w_dmrl":1,
                        "dmrl_alpha":2,
                        "dmrl_sort":True
                    },
                    optimizerStepEvery=2,
                    l2_pool_paper=True,
                    cnn_model="convnext_tiny",
                    lr=2e-5,
                    optimizer="AdamW")
train(initData,test=False)