#!/bin/bash

dataset=$1

LINK="https://zenodo.org/record/6903423/files/${dataset}"

if [[ $dataset == *.pkl ]]; then
    mkdir datasets
    wget $LINK -O datasets/${dataset}
else
    if [[ $dataset == *.tar.gz ]]; then
        echo "downloading ${dataset}..."
        wget $LINK
        tar -xzvf $dataset
    fi
fi
