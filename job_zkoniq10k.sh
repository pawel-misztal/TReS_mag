#!/bin/bash
set -x

python3 eval_zkoniq10k.py --seed 2137
python3 eval_zkoniq10k.py --seed 21372
python3 eval_zkoniq10k.py --seed 21373
python3 eval_zkoniq10k.py --seed 21374

set +x