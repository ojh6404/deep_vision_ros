#!/usr/bin/bash

git submodule update --init --recursive
sudo apt install -y python3.9 python3.9-dev python3.9-venv
python3.9 -m pip install -U numpy pip setuptools

# Retrieve the CUDA version using the nvcc command
cuda_version_full=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
cuda_major_version=$(echo $cuda_version_full | cut -d'.' -f1)

# install jax with cuda support for better performance
if [ "$cuda_major_version" == "V12" ]; then
    python3.9 -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
    python3.9 -m pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
elif [ "$cuda_major_version" == "V11" ]; then
    python3.9 -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    python3.9 -m pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "CUDA Major Version is not 11 or 12. Found version: $cuda_major_version"
    # Handle other versions or exit
fi
