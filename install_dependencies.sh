#!/bin/bash

# Create conda environment
conda create -n collapse python=3.9
conda activate collapse

# Install conda packages
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch

export TORCH=1.11.0
export CUDA=cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster-f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

conda install -c pytorch faiss-gpu

# Install GVP
git clone https://github.com/drorlab/gvp-pytorch
cd gvp-pytorch
pip install .
cd ..

# install pip packages
pip install -r requirements.txt
