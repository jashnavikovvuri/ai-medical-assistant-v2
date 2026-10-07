import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🧠",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("🧠 AI Medical Assistant")
st.subheader("Brain MRI Classification")


st.write(
    "Upload a Brain MRI image to get an AI-based classification result."
)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_ai_model():
    return load_model("resnet50_brain_mri.keras")


model = load_ai_model()


# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Brain MRI",
        use_container_width=True
    )

    if st.button("🔍 Analyze MRI"):

        with st.spinner("Analyzing MRI image..."):

            # Resize image
            img = image.resize((224, 224))

            # Convert to array
            img_array = np.array(img, dtype=np.float32)

            # Add batch dimension
            input_array = np.expand_dims(img_array, axis=0)

            # ResNet50 preprocessing
            input_array = preprocess_input(input_array)

            # Prediction
            prediction = model.predict(
                input_array,
                verbose=0
            )

            # Get predicted class
            predicted_index = np.argmax(prediction[0])

            predicted_class = class_names[predicted_index]

            confidence = (
                prediction[0][predicted_index] * 100
            )


        # -----------------------------
        # Results
        # -----------------------------

        st.success(
            f"Predicted Class: {predicted_class.capitalize()}"
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # -----------------------------
        # Class Probabilities
        # -----------------------------

        st.subheader("📊 Class Probabilities")

        for class_name, probability in zip(
            class_names,
            prediction[0]
        ):

            st.write(
                f"**{class_name.capitalize()}**: "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


# -----------------------------
# Medical Disclaimer
# -----------------------------

st.warning(
    """
⚠️ Medical Disclaimer

This application provides an AI-based screening result
for educational and project purposes only.

It is NOT a medical diagnosis and should not replace
evaluation by a qualified medical professional.
"""
)


# -----------------------------
# Footer
# -----------------------------

st.caption(
    "AI Medical Assistant | Brain MRI Classification"
)

st.caption(
    "Built using ResNet50 + TensorFlow + Streamlit"
)
