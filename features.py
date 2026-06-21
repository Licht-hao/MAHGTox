import re
import torch
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen
from mordred import Calculator, descriptors
from torch_geometric.data import Data
import json
from collections import Counter

MAX_SMILES_LEN = 128
FP_BITS_MORGAN = 1024
FP_BITS_ATOMPAIR = 2048
FP_BITS_MACCS = 167

def one_hot_encoding(value, choices):
    return [1 if value == c else 0 for c in choices]

def atom_features(atom):
    atomic_numbers = [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21,28,29,30,31,32,33,34,35,36,37,38,46,47,48,49,50,51,52,53]
    atomic_feat = one_hot_encoding(atom.GetAtomicNum(), atomic_numbers)
    degree_feat = one_hot_encoding(atom.GetTotalDegree(), list(range(5)))
    hyb_types = list(Chem.rdchem.HybridizationType.names.values())
    hyb_feat = one_hot_encoding(int(atom.GetHybridization()), hyb_types)
    chiral_types = list(Chem.rdchem.ChiralType.names.values())
    chiral_feat = one_hot_encoding(atom.GetChiralTag(), chiral_types)
    h_count_feat = one_hot_encoding(atom.GetTotalNumHs(), list(range(5)))
    aromatic_feat = [1 if atom.GetIsAromatic() else 0]
    return np.array(atomic_feat + degree_feat + hyb_feat + chiral_feat + h_count_feat + aromatic_feat, dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    bond_type_feat = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0
    ]
    conjugated_feat = [1 if bond.GetIsConjugated() else 0]
    ring_feat = [1 if bond.IsInRing() else 0]
    return np.array(bond_type_feat + conjugated_feat + ring_feat, dtype=np.float32)

def mol_to_graph(mol):
    if mol is None: return None
    n_atoms = mol.GetNumAtoms()
    node_features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([u, v])
        edge_indices.append([v, u])
        feat = bond_features(bond)
        edge_attrs.append(feat)
        edge_attrs.append(feat)
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class SMILESTokenizer:
    def __init__(self, max_len=128):
        self.max_len = max_len
        self.pattern = r"(\[[^\[\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)
        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.vocab = {t: i for i, t in enumerate(self.special_tokens)}
    def tokenize(self, smiles):
        return [token for token in self.regex.findall(smiles)]
    def build_vocab(self, smiles_list):
        counter = Counter()
        for s in smiles_list:
            counter.update(self.tokenize(s))
        for token, _ in counter.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        return self.vocab
    def encode(self, smiles):
        tokens = self.tokenize(smiles)[:self.max_len - 2]
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        return [self.vocab["<sos>"]] + ids + [self.vocab["<eos>"]]

def get_global_tokenizer(json_path):
    with open(json_path, 'r') as f:
        vocab = json.load(f)
    tokenizer = SMILESTokenizer(max_len=MAX_SMILES_LEN)
    tokenizer.vocab = vocab
    return tokenizer

def get_morgan_fingerprint(mol):
    if mol is None:
        return np.zeros(FP_BITS_MORGAN, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS_MORGAN)
    return np.array(fp, dtype=np.float32)

def get_atompair_fingerprint(mol):
    if mol is None:
        return np.zeros(FP_BITS_ATOMPAIR, dtype=np.float32)
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=FP_BITS_ATOMPAIR)
    return np.array(fp, dtype=np.float32)

def get_maccs_fingerprint(mol):
    if mol is None:
        return np.zeros(FP_BITS_MACCS, dtype=np.float32)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return np.array(fp, dtype=np.float32)

def get_fingerprint(mol):
    morgan = get_morgan_fingerprint(mol)
    atompair = get_atompair_fingerprint(mol)
    maccs = get_maccs_fingerprint(mol)
    return np.hstack([morgan, atompair, maccs]).astype(np.float32)

_MORDRED_CALCULATOR = None

def get_mordred_calculator():
    global _MORDRED_CALCULATOR
    if _MORDRED_CALCULATOR is None:
        _MORDRED_CALCULATOR = Calculator(descriptors, ignore_3D=True)
    return _MORDRED_CALCULATOR

def compute_mordred_batch(mol_list, nproc=1):
    if not mol_list:
        return np.empty((0, 0))
    calc = get_mordred_calculator()
    df = calc.pandas(mol_list, nproc=nproc, quiet=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0).astype(np.float32)
    return df.values