import torch
import numpy as np
import pickle
import json
import os
import math
from rdkit import Chem
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from models import MultiModalMAHClassifier
from features import mol_to_graph, get_fingerprint, compute_mordred_batch

DEVICE = "cpu"

def _require_file(path, kind="file"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {kind}: {path}\nPlease confirm the file exists or check the path.")
    return path

def load_model_for_dataset(model_pt_path, params_json_path, scaler_pkl_path, tokenizer, device=DEVICE):
    _require_file(params_json_path, "hyperparameter file")
    _require_file(scaler_pkl_path, "scaler file")
    _require_file(model_pt_path, "model weight file")

    with open(params_json_path, 'r') as f:
        best_params = json.load(f)
    with open(scaler_pkl_path, 'rb') as f:
        scaler = pickle.load(f)

    dummy_mol = Chem.MolFromSmiles("CC")
    graph_dim = mol_to_graph(dummy_mol).x.shape[1]
    fp_dim = 1024 + 2048 + 167 
    mordred_dim = scaler.mean_.shape[0]
    vocab_size = len(tokenizer.vocab)
    pad_idx = tokenizer.vocab["<pad>"]

    model = MultiModalMAHClassifier(
        graph_dim=graph_dim,
        fp_dim=fp_dim,
        mordred_dim=mordred_dim,
        sm_vocab_size=vocab_size,
        sm_pad_idx=pad_idx,
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        gnn_hidden=best_params['gnn_hidden'],
        gnn_heads=best_params['gnn_heads'],
        gnn_dropout=best_params['gnn_dropout'],
        trans_d_model=best_params['trans_d_model'],
        trans_nhead=best_params['trans_nhead'],
        trans_num_layers=best_params['trans_num_layers'],
        trans_dropout=best_params['trans_dropout'],
        fusion_attn_layers=best_params['fusion_attn_layers'],
        fusion_num_heads=best_params['fusion_num_heads'],
        fusion_dropout=best_params['dropout']
    ).to(device)
    state_dict = torch.load(model_pt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, best_params


def load_all_models(mutagenicity_model, mutagenicity_params, mutagenicity_scaler,
                    in_vitro_model, in_vitro_params, in_vitro_scaler,
                    in_vivo_model, in_vivo_params, in_vivo_scaler,
                    tokenizer, device=DEVICE):
    models = {}
    scalers = {}

    model_mut, scaler_mut, _ = load_model_for_dataset(
        mutagenicity_model, mutagenicity_params, mutagenicity_scaler, tokenizer, device
    )
    models['mutagenicity'] = model_mut
    scalers['mutagenicity'] = scaler_mut

    model_vitro, scaler_vitro, _ = load_model_for_dataset(
        in_vitro_model, in_vitro_params, in_vitro_scaler, tokenizer, device
    )
    models['in_vitro'] = model_vitro
    scalers['in_vitro'] = scaler_vitro

    model_vivo, scaler_vivo, _ = load_model_for_dataset(
        in_vivo_model, in_vivo_params, in_vivo_scaler, tokenizer, device
    )
    models['in_vivo'] = model_vivo
    scalers['in_vivo'] = scaler_vivo

    return models, scalers

def _empty_result(smi):
    return {
        "SMILES": smi,
        "Mutagenicity_prob": np.nan,
        "Chromosomal_Damage_in_vitro_prob": np.nan,
        "Chromosomal_Damage_in_vivo_prob": np.nan,
    }


def predict_smiles_batch(smiles_list, models, scalers, tokenizer, device=DEVICE,
                          batch_size=64, mordred_nproc=1, progress_callback=None):
    n_total = len(smiles_list)

    norm_smiles = []
    for smi in smiles_list:
        if isinstance(smi, str):
            norm_smiles.append(smi.strip())
        else:
            norm_smiles.append("")

    unique_smiles = []
    smi_to_uidx = {}
    for s in norm_smiles:
        if s and s not in smi_to_uidx:
            smi_to_uidx[s] = len(unique_smiles)
            unique_smiles.append(s)

    if not unique_smiles:
        if progress_callback is not None:
            progress_callback(0, 1)
        return [_empty_result(smi) for smi in smiles_list]

    valid_uidx = []
    mols, graphs, seqs, fps = [], [], [], []
    for uidx, smi in enumerate(unique_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        graph = mol_to_graph(mol)
        if graph is None:
            continue
        valid_uidx.append(uidx)
        mols.append(mol)
        graphs.append(graph)
        seqs.append(torch.tensor(tokenizer.encode(smi), dtype=torch.long))
        fps.append(torch.tensor(get_fingerprint(mol), dtype=torch.float32))

    n_valid = len(mols)
    unique_results = {} 

    if n_valid > 0:
        raw_mordred = compute_mordred_batch(mols, nproc=mordred_nproc)

        mordred_by_model = {}
        for name, scaler in scalers.items():
            norm = scaler.transform(raw_mordred)
            mordred_by_model[name] = torch.tensor(norm, dtype=torch.float32).to(device)

        pad_idx = tokenizer.vocab["<pad>"]
        seq_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx).to(device)

        fps_tensor = torch.stack(fps).to(device)

        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)
            sub_graphs = graphs[start:end]
            batch_graph = Batch.from_data_list(sub_graphs).to(device)
            batch_seq = seq_padded[start:end]
            batch_fp = fps_tensor[start:end]

            for name, model in models.items():
                batch_mordred = mordred_by_model[name][start:end]
                with torch.no_grad():
                    logits = model((batch_graph, batch_seq, batch_mordred, batch_fp))
                    probs = torch.sigmoid(logits).cpu().numpy()
                for i, prob in enumerate(probs):
                    uidx = valid_uidx[start + i]
                    unique_results.setdefault(uidx, {})[name] = float(prob)

            if progress_callback is not None:
                progress_callback(end, n_valid)
    else:
        if progress_callback is not None:
            progress_callback(0, 1)

    final_output = []
    for smi, s in zip(smiles_list, norm_smiles):
        uidx = smi_to_uidx.get(s)
        pred = unique_results.get(uidx, {}) if uidx is not None else {}
        final_output.append({
            "SMILES": smi,
            "Mutagenicity_prob": pred.get('mutagenicity', np.nan),
            "Chromosomal_Damage_in_vitro_prob": pred.get('in_vitro', np.nan),
            "Chromosomal_Damage_in_vivo_prob": pred.get('in_vivo', np.nan),
        })
    return final_output