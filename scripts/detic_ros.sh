#!/bin/bash

echo "Runnig Detic Ros..."
echo "Path: $1"
echo "Host: $2"
echo "Compressed: $3"
echo "Input Image: $4"
echo "Model Type: $5"
echo "Vocabulary: $6"
echo "Custom Vocabulary: $7"

if [ $6="custom" ]; then
    echo "Running custom vocabulary..."
    python3 $1/run_container.py -host $2 -mount $1/launch -name sample.launch \
    out_debug_img:=true \
    out_debug_segimg:=false \
    compressed:=$3 \
    device:=auto \
    input_image:=$4 \
    model_type:=$5 \
    vocabulary:=$6 \
    custom_vocabulary:=$7
else
    echo "Running default vocabulary..."
    python3 $1/run_container.py -host $2 -mount $1/launch -name sample.launch \
    out_debug_img:=true \
    out_debug_segimg:=false \
    compressed:=$3 \
    device:=auto \
    input_image:=$4 \
    model_type:=$5

fi
