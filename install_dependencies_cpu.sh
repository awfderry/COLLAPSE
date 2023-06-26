#!/bin/bash

# Create conda environment
#conda create -n collapse python=3.8
#conda activate collapse

# Install conda packages
conda install pytorch==1.13.1 -c pytorch

conda install pyg -c pyg
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

conda install -c pytorch faiss-cpu

# Install GVP
git clone https://github.com/drorlab/gvp-pytorch
cd gvp-pytorch
pip install .
cd ..

# install pip packages
pip install -r requirements.txt
