dataset = $1

LINK='https://zenodo.org/record/6903423/files/${1}?download=1'

if [[ $LINK == *.pkl ]]; then
    mkdir datasets
    wget $LINK -O datasets/${dataset}
else
    if [[ $LINK == *.tar.gz ]]; then
        wget $LINK
        tar -xzvf $LINK
    fi
fi