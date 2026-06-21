import os
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import torch
from io import BytesIO
from utils import load_all_models, predict_smiles_batch
from features import get_global_tokenizer

st.set_page_config(page_title="Genotoxicity Predictor", layout="wide")
st.title("🧬 MAHGTox")
st.subheader("A Multimodal Adaptive Hierarchical Network for Genotoxicity Prediction")
st.markdown("Upload files with **SMILES** column to predict three genotoxicity endpoints")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
torch.set_num_threads(1)

MODEL_FILES = {
    "tokenizer": "global_tokenizer.json",
    "mutagenicity_model": "mutagenicity_model.pt",
    "mutagenicity_params": "mutagenicity_best_hyperparams.json",
    "mutagenicity_scaler": "mutagenicity_mordred_scaler.pkl",
    "in_vitro_model": "chromosomal_damage_in_vitro_model.pt",
    "in_vitro_params": "chromosomal_damage_in_vitro_best_hyperparams.json",
    "in_vitro_scaler": "chromosomal_damage_in_vitro_mordred_scaler.pkl",
    "in_vivo_model": "chromosomal_damage_in_vivo_model.pt",
    "in_vivo_params": "chromosomal_damage_in_vivo_best_hyperparams.json",
    "in_vivo_scaler": "chromosomal_damage_in_vivo_mordred_scaler.pkl",
}


def _abs(name):
    return os.path.join(BASE_DIR, MODEL_FILES[name])

def check_required_files():
    missing = [MODEL_FILES[k] for k in MODEL_FILES if not os.path.exists(_abs(k))]
    return missing

@st.cache_resource
def load_tokenizer():
    return get_global_tokenizer(_abs("tokenizer"))

@st.cache_resource
def load_models():
    tokenizer = load_tokenizer()
    models, scalers = load_all_models(
        mutagenicity_model=_abs("mutagenicity_model"),
        mutagenicity_params=_abs("mutagenicity_params"),
        mutagenicity_scaler=_abs("mutagenicity_scaler"),
        in_vitro_model=_abs("in_vitro_model"),
        in_vitro_params=_abs("in_vitro_params"),
        in_vitro_scaler=_abs("in_vitro_scaler"),
        in_vivo_model=_abs("in_vivo_model"),
        in_vivo_params=_abs("in_vivo_params"),
        in_vivo_scaler=_abs("in_vivo_scaler"),
        tokenizer=tokenizer,
        device=DEVICE,
    )
    return models, scalers


missing_files = check_required_files()
if missing_files:
    st.error(
        "❌ missing files: \n\n"
        + "\n".join(f"- {f}" for f in missing_files)
    )
    st.stop()


uploaded_file = st.file_uploader("📂 Upload Excel files (.xlsx)", type=["xlsx"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    cache = st.session_state.get("prediction_cache")

    if cache is not None and cache.get("file_hash") == file_hash:
        output = cache["output"]
        excel_bytes = cache["excel_bytes"]
        st.write(f"Total {cache['n_total']} rows, with {cache['n_unique']} unique valid SMILES")
        st.success("✅ Prediction completed")
    else:
        try:
            df = pd.read_excel(BytesIO(file_bytes))
        except Exception as e:
            st.error(f"❌ Failed to read Excel file: {e}")
            st.stop()

        if "SMILES" not in df.columns:
            st.error("❌ The file must contain a 'SMILES' column")
            st.stop()

        if len(df) == 0:
            st.warning("⚠️ The file contains no data rows")
            st.stop()

        smiles_list = df["SMILES"].tolist()
        n_total = len(smiles_list)
        n_unique = len(set(s.strip() for s in smiles_list if isinstance(s, str) and s.strip()))
        st.write(f"Total {n_total} rows, with {n_unique} unique valid SMILES, predicting...")

        try:
            with st.spinner("Loading models (first run may take some time)..."):
                models, scalers = load_models()
                tokenizer = load_tokenizer()
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Preprocessing molecules and computing features (Mordred descriptors may take a while)...")

        def _update_progress(done, total):
            pct = int(done / total * 100) if total else 100
            progress_bar.progress(min(max(pct, 0), 100))
            status_text.text(f"Predicting... {done}/{total} unique molecules")

        try:
            results = predict_smiles_batch(
                smiles_list,
                models,
                scalers,
                tokenizer,
                device=DEVICE,
                batch_size=64,
                mordred_nproc=1,
                progress_callback=_update_progress,
            )
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.stop()

        progress_bar.progress(100)
        status_text.text("Prediction completed!")

        result_df = pd.DataFrame(results)

        output = pd.concat(
            [df.reset_index(drop=True), result_df.drop(columns=["SMILES"]).reset_index(drop=True)],
            axis=1,
        )
        output['Potential_Genotoxicity'] = (
            (output['Mutagenicity_prob'] > 0.5) |
            (output['Chromosomal_Damage_in_vitro_prob'] > 0.5) |
            (output['Chromosomal_Damage_in_vivo_prob'] > 0.5)
        )

        n_invalid = int(output["Mutagenicity_prob"].isna().sum())
        if n_invalid > 0:
            st.warning(f"⚠️ {n_invalid} rows have SMILES that cannot be parsed or molecular graph cannot be constructed; corresponding predictions are NaN")

        st.success("✅ Prediction completed")

        to_excel = BytesIO()
        with pd.ExcelWriter(to_excel, engine="openpyxl") as writer:
            output.to_excel(writer, index=False)
        excel_bytes = to_excel.getvalue()

        st.session_state["prediction_cache"] = {
            "file_hash": file_hash,
            "output": output,
            "excel_bytes": excel_bytes,
            "n_total": n_total,
            "n_unique": n_unique,
        }

    st.dataframe(output)

    st.download_button(
        label="📥 Download predictions (Excel)",
        data=excel_bytes,
        file_name="toxicity_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )