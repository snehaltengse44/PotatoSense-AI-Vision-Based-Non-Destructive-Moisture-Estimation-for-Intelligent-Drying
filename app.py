import streamlit as st
from inference import MoisturePredictor
from potato_gate import PotatoGate

st.set_page_config(page_title="Potato Moisture Predictor", page_icon="🥔", layout="centered")

@st.cache_resource
def load_predictor():
    return MoisturePredictor()

@st.cache_resource
def load_gate():
    return PotatoGate(reference_embeddings_path="tuned_model/reference_embeddings.npy", threshold=0.65)

predictor = load_predictor()
gate = load_gate()

st.title("🥔 Potato Moisture Content Predictor")
st.write("Upload a potato slice image to estimate moisture content (%).")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded image", use_column_width=True)

    if st.button("Predict Moisture Content"):
        with st.spinner("Analyzing image..."):
            try:
                feats = predictor.extract_features_raw(image_bytes)
                gate_result = gate.check(feats)
            except Exception as e:
                st.error(f"Image processing failed: {e}")
                feats = None

        if feats is not None:
            if not gate_result["is_valid"]:
                st.error(
                    f"⚠️ Invalid image — this doesn't look like a potato slice "
                    f"(similarity: {gate_result['similarity']:.0%}). "
                    f"Please upload a clear photo of a cut potato."
                )
            else:
                result = predictor.predict_from_features(feats)
                st.success(f"Predicted Moisture Content: **{result}%**")

st.markdown("---")
st.caption(f"Model: resnet50_svr_linear_tuned · {predictor.meta.get('feature_type', '')}")