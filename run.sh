#!/bin/bash
set -x
echo "startujemy"

python3 org_best_v4_1.py -jcp /home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints/Tres_clive_1140551136897037627.json -cp /home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints/Tres_clive_1140551136897037627_last.pth
python3 org_best_v4_2.py
python3 org_best_v4_3.py
python3 org_best_v4_4.py
python3 org_best_v4_5.py

set +x