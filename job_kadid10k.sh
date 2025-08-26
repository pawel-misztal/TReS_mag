#!/bin/bash
set -x

python3 eval_kadid10k.py --seed 2137
python3 eval_kadid10k.py --seed 21372
python3 eval_kadid10k.py --seed 21373

set +x