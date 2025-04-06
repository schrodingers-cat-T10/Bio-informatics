import streamlit as st
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Flatten, BatchNormalization, Dropout
from lime.lime_tabular import LimeTabularExplainer
import subprocess
import os
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
import google.generativeai as genai


def ocr(file):
    gemini_api_key = "AIzaSyCuD4imF8Ptr5rlqd-A5wSBwDJXN_n6y8I"  
    genai.configure(api_key=gemini_api_key)
    img = Image.open(file)
    prompt = (
        "you are explainable ai agent , you will be provided with lime graph picture , please check it and gimme a perfect inference , im a bioinformatics research "
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([prompt, img])
    return response.text


def target_frame(target_id):
    search_target = new_client.target
    target = search_target.search(target_id)
    return pd.DataFrame.from_dict(target)

def chembl_id(chembl_id):
    activity = new_client.activity
    response = activity.filter(target_chembl_id=chembl_id).filter(standard_type="IC50")
    return pd.DataFrame.from_dict(response)

def data_preprocessing(pre_data):
    pre_data = pre_data[pre_data["standard_value"].notna()]
    pre_data = pre_data[pre_data["canonical_smiles"].notna()]
    pre_data = pre_data.drop_duplicates(["canonical_smiles"])
    pre_data = pre_data[["molecule_chembl_id", "canonical_smiles", "standard_value"]].dropna()
    pre_data["status"] = pre_data["standard_value"].apply(
        lambda x: "active" if float(x) >= 10000 else "non-active" if float(x) <= 1000 else "intermediate"
    )
    pre_data = pre_data[pre_data["status"] != "intermediate"]
    smiles_data = pre_data[["molecule_chembl_id", "canonical_smiles"]]
    smiles_data.to_csv("molecules.smi", index=False)
    return smiles_data, pre_data

def conversion():
    df = pd.read_csv("molecules.smi")
    with open("molecules.smi", "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row['canonical_smiles']} {row['molecule_chembl_id']}\n")

def run_padel_script_and_get_descriptors():
    try:
        subprocess.run([
            "java", "-Xms1G", "-Xmx1G", "-Djava.awt.headless=true", "-jar",
            "PaDEL-Descriptor/PaDEL-Descriptor.jar",
            "-removesalt", "-standardizenitro", "-fingerprints",
            "-descriptortypes", "PaDEL-Descriptor/PubchemFingerprinter.xml",
            "-dir", ".", "-file", "descriptors_output.csv"
        ], check=True)
        return pd.read_csv("descriptors_output.csv")
    except Exception as ex:
        st.error(f"PaDEL failed: {ex}")
        return None


def build_model(data, descriptors):
    y = data["status"]
    x = descriptors.select_dtypes(include=[np.number])
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(x_train.shape[1], 1)),
        BatchNormalization(), Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(), Dropout(0.2),
        Bidirectional(LSTM(32)),
        BatchNormalization(), Dropout(0.2),
        Flatten(), Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(2, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)

    return model, history.history, x_test, y_test, encoder


st.set_page_config(layout="wide")
st.title("🧪 Drug Discovery with Explainable AI")

if "stage" not in st.session_state:
    st.session_state.stage = "input"


if st.session_state.stage == "input":
    with st.form("target_form"):
        target_input = st.text_input("Enter a target name (e.g., Acetylcholinesterase)")
        submitted = st.form_submit_button("Search")

    if submitted and target_input:
        df_targets = target_frame(target_input)
        if df_targets.empty:
            st.warning("No targets found.")
        else:
            st.session_state.df_targets = df_targets
            st.session_state.stage = "select"


if st.session_state.stage == "select":
    df_targets = st.session_state.df_targets
    st.subheader("🎯 Matching Targets")
    st.dataframe(df_targets[["pref_name", "organism", "target_chembl_id"]])
    selected_id = st.selectbox("Select a ChEMBL Target ID", df_targets["target_chembl_id"])
    if st.button("Run Pipeline"):
        st.session_state.selected_id = selected_id
        st.session_state.stage = "processing"


if st.session_state.stage == "processing":
    st.info("Running drug discovery pipeline...")
    activity_data = chembl_id(st.session_state.selected_id)
    if activity_data.empty:
        st.error("No activity data.")
        st.session_state.stage = "input"
    else:
        smiles_data, labeled_data = data_preprocessing(activity_data)
        conversion()
        descriptors = run_padel_script_and_get_descriptors()
        if descriptors is not None:
            mdl, history, x_test, y_test, encoder = build_model(labeled_data, descriptors)
            st.session_state.model = mdl
            st.session_state.history = history
            st.session_state.x_test = x_test
            st.session_state.y_test = y_test
            st.success("✅ Model trained!")
            st.session_state.stage = "explain"


if st.session_state.stage == "explain":
    st.line_chart({
        "Train Accuracy": st.session_state.history["accuracy"],
        "Validation Accuracy": st.session_state.history["val_accuracy"]
    })

    idx = st.selectbox("Select sample index to explain", list(range(len(st.session_state.x_test))))

    if "lime_img_path" not in st.session_state or st.session_state.get("lime_idx") != idx:
        explainer = LimeTabularExplainer(
            training_data=st.session_state.x_test.reshape(
                st.session_state.x_test.shape[0], st.session_state.x_test.shape[1]),
            feature_names=[f"f{i}" for i in range(st.session_state.x_test.shape[1])],
            class_names=["non-active", "active"],
            mode="classification"
        )
        explanation = explainer.explain_instance(
            st.session_state.x_test[idx].reshape(-1),
            st.session_state.model.predict,
            num_features=10
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig = explanation.as_pyplot_figure()
            fig.savefig(tmp.name)
            plt.close(fig)
            st.session_state.lime_img_path = tmp.name
            st.session_state.lime_idx = idx

    st.image(st.session_state.lime_img_path, caption=f"LIME Explanation for sample {idx}")

    if st.button("🔍 Explain with Gemini"):
        gemini_result = ocr(st.session_state.lime_img_path)
        st.markdown(f"**Gemini Insight:**\n\n{gemini_result}")
