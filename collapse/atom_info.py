bb_atoms = ['N', 'CA', 'C', 'O']

aa = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]
aa_abbr = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
]
letter_to_aa = lambda x: dict(zip(aa_abbr, aa)).get(x, 'X')
aa_to_letter = lambda x: dict(zip(aa, aa_abbr)).get(x, 'X')

aa_to_label = lambda x: {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'E': 5,
    'Q': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19
}.get(x, 20)

prosite_residues = {
    'EGF_1': 'CYS',
    'TRYPSIN_SER': 'SER',
    'RNASE_PANCREATIC': 'LYS',
    'EF_HAND_1': 'ASP',
    'IG_MHC': 'CYS',
    'PROTEIN_KINASE_TYR': 'ASP',
    'TRYPSIN_HIS': 'HIS',
    'INSULIN': 'CYS',
    'PROTEIN_KINASE_ST': 'ASP',
    'ADH_SHORT': 'TYR'
}

res_group_dict = {
    'HIS': 'positive', 'LYS': 'positive', 'ARG': 'positive',
    'ASP': 'negative', 'GLU': 'negative',
    'SER': 'small_polar', 'THR': 'small_polar', 'ASN': 'small_polar', 'GLN': 'small_polar',
    'ALA': 'small_hydrophobic', 'VAL': 'small_hydrophobic', 'LEU': 'small_hydrophobic', 'ILE': 'small_hydrophobic', 'MET': 'small_hydrophobic',
    'PHE': 'large_hydrophobic', 'TYR': 'large_hydrophobic', 'TRP': 'large_hydrophobic',
    'PRO': 'unique', 'GLY': 'unique',
    'CYS': 'cysteine'
}

res_key_atom_dict = {
    'ILE': [["CB"]],
    'GLN': [["OE1", "CD", "NE2"]],
    'GLY': [["CA"]],
    'GLU': [["OE1", "CD", "OE2"]],
    'CYS': [["SG"]],
    'HIS': [["NE2", "ND1"]],
    'SER': [["OG"]],
    'LYS': [["NZ"]],
    'PRO': [["N", "CA", "CB", "CD", "CG"]],
    'ASN': [["OD1", "CG", "ND2"]],
    'VAL': [["CB"]],
    'THR': [["OG1"]],
    'ASP': [["OD1", "CG", "OD2"]],
    'TRP': [["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"], ["NE1"]],
    'PHE': [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    'ALA': [["CB"]],
    'MET': [["SD"]],
    'LEU': [["CB"]],
    'ARG': [["CZ"]],
    'TYR': [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], ["OH"]]
}

abbr_key_atom_dict = {aa_to_letter(k): v for k, v in res_key_atom_dict.items()}

res_atom_valence = {
    'ALA': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH3',
        'O': 'O',
        'OXT': 'OH',
        'N': 'NH'
    },
    'CYS': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'O': 'O',
        'OXT': 'OH',
        'N': 'NH',
        'SG': 'SH'
    },
    'ASP': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'C',
        'O': 'O',
        'OD1': 'O',
        'OD2': 'OH',
        'OXT': 'OH',
        'N': 'NH',
    },
    'GLU': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CH',
        'CD': 'C',
        'O': 'O',
        'OE1': 'O',
        'OE2': 'OH',
        'OXT': 'OH',
        'N': 'NH',
    },
    'PHE': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'CD2': 'CPi',
        'CE1': 'CPi',
        'CE2': 'CPi',
        'CZ': 'CPi',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
    },
    'GLY': {
        'C': 'C',
        'CA': 'CH2',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH'
    },
    'HIS': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CE1': 'CPi',
        'CD2': 'CPi',
        'N': 'NH',
        'ND1': 'N',
        'ND2': 'NH2', 
        'O': 'O',
        'OXT': 'OH',
    },
    'ILE': {
        'C': 'C',
        'CA': 'CH',
        'CB': 'CH',
        'CG1': 'CH2',
        'CG2': 'CH3',
        'CD1': 'CH3',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH'
    },
    'LYS': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2',
        'CE': 'CH2',
        'NZ': 'NH2'
    },
    'LEU': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH',
        'CD1': 'CH3',
        'CD2': 'CH3',
    },
    'MET': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'SD': 'S',
        'CE': 'CH3'
    },
    'ASN': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'C',
        'OD1': 'O',
        'ND2': 'NH2'
    },
    'PRO': {
        'C': 'C',
        'CA': 'CH',
        'N': 'N',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2'
    },
    'GLN': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'C',
        'OE1': 'O',
        'NE2': 'NH2'
    },
    'ARG': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CH2',
        'CD': 'CH2',
        'NE': 'NH',
        'CZ': 'C',
        'NH1': 'NH',
        'NH2': 'NH2'
    },
    'SER': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'OG': 'OH',
    },
    'THR': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH',
        'OG1': 'OH',
        'CG2': 'CH3'
    },
    'VAL': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH',
        'CG1': 'CH3',
        'CG2': 'CH3',
    },
    'TRP': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'NE1': 'NH',
        'CD2': 'CPi',
        'CE2': 'CPi',
        'CZ2': 'CPi',
        'CH2': 'CPi',
        'CZ3': 'CPi',
        'CE3': 'CPi'
    },
    'TYR': {
        'C': 'C',
        'CA': 'CH',
        'N': 'NH',
        'O': 'O',
        'OXT': 'OH',
        'CB': 'CH2',
        'CG': 'CPi',
        'CD1': 'CPi',
        'CE1': 'CPi',
        'CZ': 'CPi',
        'CE2': 'CPi',
        'CD2': 'CPi',
        'OH': 'OH'
    }

}

cath_toplevel_dict = {'1': 'mainly alpha', '2': 'mainly beta', '3': 'alpha beta', '4': 'few sec. structures', '6': 'special'}

ss_label_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
label_ss_dict = {v: k for k, v in ss_label_dict.items()}
ss_name_dict = {'H': 'alpha helix', 'B': 'beta bridge', 'E': 'beta sheet', 'G': '3-10 helix', 'I': 'pi helix', 'T': 'H-bonded turn', 'S': 'bend', '-': 'coil'}
label_name_dict = {v: ss_name_dict[k] for k, v in ss_label_dict.items()}
