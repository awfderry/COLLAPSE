#!/bin/bash

# Create conda environment
#conda create -n collapse3 python=3.9
#conda activate collapse3

# Install conda packages
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg

conda install -c pytorch faiss-gpu

# Install GVP
git clone https://github.com/drorlab/gvp-pytorch
cd gvp-pytorch
pip install .
cd ..

# install pip packages
pip install -r requirements.txt

