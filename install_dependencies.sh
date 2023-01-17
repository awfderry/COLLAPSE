#!/bin/bash

# Create conda environment
conda create -n collapse_alp3 python=3.9
source activate collapse_alp3

# Install conda packages

pip install numpy
ml gcc/10.1.0
pip install torch==1.11.0+cu113  --extra-index-url https://download.pytorch.org/whl/cu113

export TORCH=1.11.0
export CUDA=cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

conda install -c pytorch faiss-gpu

# Install GVP
git clone https://github.com/drorlab/gvp-pytorch
cd gvp-pytorch
pip install .
cd ..

# install pip packages
pip install -r requirements.txt

