#!/usr/bin/bash

pip3 install gdown # Install gdown to download the model
git submodule update --init --recursive
sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install numpy==1.26.4 psutil==5.9.8
python3.9 -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
CUDA_ARCH="8.0;8.6" python3.9 -m pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
