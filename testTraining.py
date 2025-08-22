from helpersmag.initData import InitData
from trainModel import main as train

initData = InitData(__file__,"Tres", epoch_count=5, eval_every_epoch=1)

train(initData,3)