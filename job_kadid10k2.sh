#!/bin/bash
set -x

python3 eval_kadid10k.py --seed 21374
python3 eval_kadid10k.py --seed 21375
python3 eval_kadid10k.py --seed 21376

set +x