#!/bin/bash

wandb sweep sweep.yaml &> output.txt
sweep_id=$(cat output.txt | awk '/ ID: /{print $6}')
wandb agent -p "vector-fusion" -e "aiis-chair" --count 10 "$sweep_id"