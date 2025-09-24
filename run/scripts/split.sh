#!/bin/bash
whereis python

echo "use python file: $1"
echo "use agent: $2"
echo "use split: $3"
echo "use CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

while true
do
    if [ -z "$API_PORT" ]; then
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $1 --agent $2 --split $3
    else
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES API_PORT=$API_PORT python $1 --agent $2 --split $3
    fi

done