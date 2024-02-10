#!/usr/bin/bash

pip3 install gdown # Install gdown to download the model
git submodule update --init --recursive
sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install -U numpy
if [ "$CUDA_VERSION" == "12.1.1" ]; then
    python3.9 -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" == "11.3.1" ]; then
    python3.9 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
else
    echo "CUDA_VERSION: ${CUDA_VERSION} is not set or supported. Please check your CUDA_VERSION environment variable is set properly like 11.3.1 or 12.1.1"
fi
