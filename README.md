# COLLAPSE
COLLAPSE (COmpressed Latents Learned from Aligned Protein Structural Environments) is a representation learning method for protein structural and functional sites, as described in Derry et al. (2022). This repo contains all package functionality as well as scripts for functional site search and annotation, pre-training, and transfer learning on Prosite and ATOM3D datasets. For more details on COLLAPSE, please see our [preprint]().

The repo is organized as follows:

  - `./collapse`: package source code and core functionality
  - `./scripts`: Python scripts and notebooks used in our preprint.
  - `./data`: datasets, model checkpoints, etc.

<p align="center"><img src="collapse_schematic.png" width="500"></p>

## Requirements

The COLLAPSE package requires the following dependencies, tested with the following versions. We recommend installing in a Conda environment.
- python==3.7.12
- torch==1.9.1
- torch_geometric=2.0.4
- torch_scatter=2.0.9
- numpy=1.21.5
- pandas=1.3.5
- scipy=1.7.3
- atom3d=0.2.5
- biopython=1.79

COLLAPSE also requires the Geometric Vector Perceptron (GVP), which can be installed from the following repo: https://github.com/drorlab/gvp-pytorch.

Scripts may require additional dependencies, which may be installed using conda or pip as needed.


## Installation

Install the package using pip:

```
pip install .
```


## Usage

Here we provide usage examples for several applications of COLLAPSE.

### Embed all residues in a PDB file

To embed all residues in a single structure using COLLAPSE, use the following lines of code. 
Here, `PDB_FILE` is the path to the PDB file containing the structure to be embedded, and `DEVICE` specifies where you want the embedding to run: `cpu` (default) or `cuda`
In this example, we only embed chain "A" and include all heteroatoms (ligands, ions, and cofactors).

```
from collapse import process_pdb, initialize_model, embed_protein

# Create model and load default parameters (pre-trained COLLAPSE model)
model = initialize_model(device=DEVICE)
# Load PDB file and pre-process to dataframe representation
atom_df = process_pdb(PDB_FILE, chain='A')
# Embed protein, returning dictionary of embeddings and metadata
emb_data = embed_protein(atom_df, model, include_hets=True, device=DEVICE)
```

The output of `embed_protein` is a dictionary containing the following data:
- `embeddings`: A numpy matrix (N x 512) containing embeddings of all N residues in protein.
- `resids`: List of length N containing residue IDs for each embedding, (e.g. A412, K23)
- `chains`: List of length N containing chain IDs for each embedding
- `confidence`: List of length N containing per-residue pLDDT scores (AlphaFold structures only)

### Embed entire dataset of PDB files

To embed all residues of all structures in a directory of PDB files, use the following script. 
`PDB_DIR` is the root directory of all the PDB files to be processed, possibly containing subdirectories. Accepted formats include `pdb`,  `pdb.gz`, and `cif`.
`OUT_DIR` is the location of the processed dataset.

```
python embed_pdb_dataset.py PDB_DIR OUT_DIR --filetype pdb
```

This script produces an embedding dataset in the LMDB format, allowing for compressed, fast, random access to all elements in the database, in which data is stored in a key-value format. Each element of the dataset produced by `embed_pdb_dataset.py` has the same keys as the outpute of `embed_protein` (see above), in addition to the following data from the initial PDB file:
  - `id`: The original PDB filename
  - `atoms`: The original PDB data, in [ATOM3D dataframe format](https://atom3d.readthedocs.io/en/latest/data_formats.html#the-atoms-dataframe)

To load this dataset in a Pytorch-style dataset format, you can use ATOM3D:

```
from atom3d.datasets import load_dataset
dataset = load_dataset(OUT_DIR, 'lmdb')
```

Additional arguments are:
- `--checkpoint` (default `./data/checkpoints/collapse_base.pt`): To specify a different checkpoint with pre-trained model parameters.
- `--split_id` (default 0): For processing dataset in chunks (useful for very large datasets), specify the split index. Must be less than `num_splits`.
- `--num_splits` (default 1): For processing dataset in chunks (useful for very large datasets), total number of splits.

If processing in chunks, each chunk of the processed dataset is stored in a `tmp_` directory. To combine these into a full processed dataset, you can use the following script from ATOM3D:

```
python -m atom3d.datasets.scripts.combine_lmdb OUT_DIR/tmp_* OUT_DIR/full
```

### Iterative search of functional site against PDB database

```
python search_site.py [params]
```

### Annotate structure using functional site database


To annotate chains A and B in the structure stored in PDB_FILE, use the following command. The output will be a printed summary of the functional sites detected and the corresponding residues. You can also supply more than one PDB file to be annotated, each separated by a space. By default, the functional site database contains conserved residues from Prosite and the Catalytic Site Atlas (CSA).

```
python annotate_pdb.py PDB_FILE --chains AB
```

Additional arguments are:
- `--checkpoint` (default `./data/checkpoints/collapse_base.pt`): To specify a different checkpoint with pre-trained model parameters.
- `--db` (default `.data/full_site_db_stats.pkl`): Functional site embedding database to use, by default a combination of Prosite and CSA. Any database can be used as long as it is in the same pickled key-value format (see below).
- `--cutoff` (default 1e-4): Empirical p-value cutoff for selecting residues with significant similarity to a functional site. Reducing this will increase precision at the cost of potentially decreasing sensitivity.
- `--site_cutoff` (default 1e-4): Empirical p-value cutoff for selecting sites with significant similarity for mutual best-hits criterion. Reducing this will increase precision at the cost of potentially decreasing sensitivity.
- `--filetype` (default `pdb`): File type of input files (must all be the same filetype).
- `--include_hets`: Flag indicating whether to include heteroatoms, such as ligands, ions, and cofactors, in the input PDB. Hydrogens and waters are always removed.
- `--verbose`: Flag indicating whether to print all PDB and residue hits in results. By default, only the total number of PDBs is shown.

## Downloading datasets

Datasets are hosted on Zenodo. 

## License

This project is licensed under the [MIT license](LICENSE)

## References

If you use COLLAPSE, please cite our preprint:

> 