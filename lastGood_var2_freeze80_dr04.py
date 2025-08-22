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
                    ffn_dropout=0.4,
                    mhsa_dropout=0.4,
                    freeze_cnn_for_epochs=80)
train(initData)