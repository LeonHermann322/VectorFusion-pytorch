#!/bin/bash

guidance_scale=${2:2}
grad_scale=${3:2}
PARAMS="skip_live=True, ${guidance_scale}, ${grad_scale}"
python3 $1 -c vectorfusion.yaml -update="${PARAMS}" -pt "${PROMPT}" -save_step 10 -respath /tmp/workdir -d 8888 --project-name "${WANDB_PROJECT}" --download