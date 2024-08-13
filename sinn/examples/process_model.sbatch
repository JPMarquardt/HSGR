#!/bin/bash
/Users/johnmarquardt/anaconda3/etc/profile.d/conda.sh
conda activate colloids

PTH="/models/24-08-10"
NAME="Alignn-k17-L4-spg22-n7"

python EVALUATE.py $PTH $NAME
python 