"""Streamlit deployment for the CIFAR-10 KNN image classifier.

Training lives only in Image_based_Product_Classification.ipynb, which saves the
trained model to disk. Run the notebook first, then:  streamlit run app_knn.py
"""
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image

# ---- Set this to wherever your saved model is (.pkl or .joblib) -------------
MODEL_FILE = "2026-2027/Computer Vision/Image Based Product Classification/Image Based Product Classification.pkl"
# -----------------------------------------------------------------------------

# Only used if the saved file contains just the model (no class names)
DEFAULT_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Image Product Classifier", page_icon="🖼️")


def find_model():
    """Look for the model next to this script, in the launch folder, then by file name."""
    p = Path(MODEL_FILE)
    candidates = [p] if p.is_absolute() else [HERE / p, Path.cwd() / p]
    candidates += sorted(HERE.glob("*.pkl")) + sorted(HERE.glob("*.joblib"))
    for c in candidates:
        if c.is_file():
            return c, candidates
    return None, candidates


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    obj = joblib.load(path)  # also reads plain pickle files
    if isinstance(obj, dict):  # {"model": knn, "class_names": [...]}
        return obj["model"], obj.get("class_names", DEFAULT_CLASSES)
    return obj, DEFAULT_CLASSES  # the KNN model saved directly


def preprocess(img: Image.Image) -> np.ndarray:
    """Same input format the model was trained on: 32x32 RGB, scaled to 0-1, flattened."""
    arr = np.asarray(img.convert("RGB").resize((32, 32)), dtype=np.float64) / 255.0
    return arr.reshape(1, -1)


st.title("🖼️ Image-based Product Classification")
st.caption("K-Nearest Neighbors (k=5) trained on CIFAR-10. Upload an image to classify it.")

model_path, tried = find_model()
if model_path is None:
    st.error("Saved model not found. Places I looked:")
    st.code("\n".join(str(c) for c in tried))
    st.info("Put the model file next to `app_knn.py`, or set `MODEL_FILE` at the top of the script "
            "to its full path.")
    st.stop()

knn, class_names = load_model(model_path)
st.sidebar.success(f"Model loaded: {model_path.name}")

files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

for f in files or []:
    img = Image.open(f)
    x = preprocess(img)
    pred = int(knn.predict(x)[0])
    proba = knn.predict_proba(x)[0]  # share of the 5 neighbours voting for each class

    left, right = st.columns([1, 2])
    left.image(img, caption=f.name, width="stretch")
    with right:
        st.subheader(class_names[pred].title())
        st.write(f"Agreement among neighbours: **{proba[pred]:.0%}**")
        top = np.argsort(proba)[::-1][:3]
        st.bar_chart({class_names[i]: float(proba[i]) for i in top}, horizontal=True)
    st.divider()

with st.expander("About this model"):
    st.write("Classes: " + ", ".join(class_names))
    st.write("Images are resized to 32×32, so low-resolution, centred, single-object photos work best. "
             "KNN on raw pixels is a simple baseline (roughly 30–35% accuracy on CIFAR-10), "
             "so treat predictions as indicative, not definitive.")
