#!/bin/bash
set -x

python3 eval_tid2013.py --seed 21378
python3 eval_tid2013.py --seed 21379
python3 eval_tid2013.py --seed 213710

set +x