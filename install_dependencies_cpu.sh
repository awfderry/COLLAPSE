#!/bin/bash

# Create conda environment
#conda create -n collapse python=3.8
#conda activate collapse

# Install conda packages
conda install pytorch==1.11.0 -c pytorch

export TORCH=1.11.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
pip install torch-geometric

conda install -c pytorch faiss-cpu

# Install GVP
git clone https://github.com/drorlab/gvp-pytorch
cd gvp-pytorch
pip install .
cd ..

# install pip packages
pip install -r requirements.txt
