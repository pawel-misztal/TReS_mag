from helpersmag.initData import InitData
from trainModel import main as train

initData = InitData(__file__,
                    "Tres", 
                    epoch_count=150, 
                    eval_every_epoch=10,
                    mhsa_add_pose_everyLayer=True,
                    normalize_before=False,
                    normalize_pos_encode=False,
                    ffn_extraDropout=False,
                    ffn_dropout=0.5,
                    mhsa_dropout=0.5,
                    ffn_size=1024, #49 size of multi head attention, times 4, like in oryginal paper
                    optimizer="AdamW",
                    loss_fn="MAExDynamicMarginRankingLoss",
                    loss_weights={
                        "w_mae":1,
                        "w_dmrl":1,
                        "dmrl_alpha":2,
                        "dmrl_sort":True
                    }
                    ,freeze_cnn_for_epochs=80)
train(initData)