#!/bin/bash
set -x

python3 eval_tid2013.py --seed 2137
python3 eval_tid2013.py --seed 21371
python3 eval_tid2013.py --seed 21372
python3 eval_tid2013.py --seed 21373
python3 eval_tid2013.py --seed 21374
python3 eval_tid2013.py --seed 21376
python3 eval_tid2013.py --seed 21377

set +x