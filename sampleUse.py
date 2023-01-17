from collapse import process_pdb, initialize_model, embed_protein
import pickle 
import numpy as np

DEVICE='cuda'
# Create model and load default parameters (pre-trained COLLAPSE model)
model = initialize_model(device=DEVICE)
# Load PDB file and pre-process to dataframe representation
atom_df = process_pdb(PDB_FILE, chain='A')
# Embed protein, returning dictionary of embeddings and metadata
emb_data = embed_protein(atom_df, model, include_hets=True, device=DEVICE)

np.save('../emb_data_dictionary.npy', emb_data)
#https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file 

with open('../emb_data_dictionary.pkl', 'wb') as f:
    pickle.dump(emb_data, f)
