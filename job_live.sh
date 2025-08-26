#!/bin/bash
set -x

python3 eval_live.py --seed 2137
python3 eval_live.py --seed 21371
python3 eval_live.py --seed 21372
python3 eval_live.py --seed 21373
python3 eval_live.py --seed 21374

set +x