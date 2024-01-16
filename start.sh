#!/bin/bash

wandb sweep sweep.yaml &> output.txt
sweep_id=$(cat output.txt | awk '/ ID: /{print $6}')
wandb agent -p "${WANDB_PROJECT}" -e "${WANDB_ENTITY}" --count 10 "$sweep_id"