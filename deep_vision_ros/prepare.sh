#!/usr/bin/bash

pip3 install gdown # Install gdown to download the model
git submodule update --init --recursive
sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install numpy==1.26.4 psutil==5.9.8
python3.9 -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python3.9 -m pip install -r requirements.txt
python3.9 -m pip install -e VisionAnything
