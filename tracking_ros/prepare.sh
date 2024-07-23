#!/usr/bin/bash

pip3 install gdown # Install gdown to download the model
git submodule update --init --recursive
sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install numpy==1.26.4 psutil==5.9.8
python3.9 -m pip install torch torchvision
