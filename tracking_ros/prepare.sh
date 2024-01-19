#!/usr/bin/bash

git submodule update --init --recursive
sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install -U numpy
python3.9 -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
