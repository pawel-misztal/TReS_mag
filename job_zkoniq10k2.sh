#!/bin/bash
set -x

python3 eval_zkoniq10k.py --seed 21379
python3 eval_zkoniq10k.py --seed 213710

set +x