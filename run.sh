#!/bin/bash

PROGRAM=$1
PARAMS=""

shift

for arg do
    trimmed=${arg:2}
    PARAMS="${PARAMS}, ${trimmed}"
done

PARAMS=${PARAMS:2}

echo "--------"
echo $PROGRAM
echo "with"
echo $PARAMS
echo "--------"

python3 $PROGRAM -c vectorfusion.yaml -update="${PARAMS}" -pt "${PROMPT}" -save_step 10 -respath /tmp/workdir -d 8888 --use_wandb --download