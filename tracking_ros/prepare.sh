#!/usr/bin/bash

pip3 install gdown # Install gdown to download the model
git submodule update --init --recursive
sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install numpy==1.25.2 psutil==5.9.8
if [ "$CUDA_VERSION" == "12.1.1" ]; then
    python3.9 -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" == "11.3.1" ]; then
    python3.9 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
else
    echo "CUDA_VERSION: ${CUDA_VERSION} is not set or supported. Please check your CUDA_VERSION environment variable is set properly like 11.3.1 or 12.1.1"
fi

# MaskDINO
cd MaskDINO/maskdino/modeling/pixel_decoder/ops && python3.9 setup.py build install --user && cd ../../../../
