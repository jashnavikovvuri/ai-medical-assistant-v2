import streamlit as st
import numpy as np
from PIL import Image
from google import genai
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Medical Assistant")
st.subheader("Brain MRI Classification")

st.write(
    "Upload a Brain MRI image to get an AI-based classification result."
)


# Generate educational explanation using Gemini
def generate_ai_explanation(predicted_class, confidence):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are an educational medical information assistant.

An image classification model predicted: {predicted_class}.
Model confidence score: {confidence:.2f}%.

Explain this class in simple English for educational purposes.
Explain that the confidence score is not a medical probability
and does not confirm a diagnosis. Recommend review by a
qualified radiologist. Do not infer any findings beyond the
provided prediction. Keep the explanation concise.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text

        return "No explanation was returned. Please try again."

    except Exception as e:
        st.error(
            f"GenAI error: {type(e).__name__}: {e}"
        )
        return "GenAI explanation is temporarily unavailable."


# Load the trained MRI classification model
@st.cache_resource
def load_ai_model():
    return load_model("resnet50_brain_mri.keras")


model = load_ai_model()

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# Upload MRI image
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Brain MRI",
            use_container_width=True
        )

        if st.button("🔍 Analyze MRI"):
            with st.spinner("Analyzing MRI image..."):
                img = image.resize((224, 224))
                img_array = np.array(
                    img,
                    dtype=np.float32
                )

                input_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                input_array = preprocess_input(input_array)

                prediction = model.predict(
                    input_array,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(prediction[0])
                )

                predicted_class = class_names[predicted_index]

                confidence = (
                    float(prediction[0][predicted_index]) * 100
                )

            # Display classification result
            st.success(
                f"Predicted Class: {predicted_class.capitalize()}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            # Generate GenAI explanation
            st.subheader("✨ GenAI Explanation")

            with st.spinner(
                "Generating educational explanation..."
            ):
                explanation = generate_ai_explanation(
                    predicted_class,
                    confidence
                )

            st.write(explanation)

            # Display class probabilities
            st.subheader("📊 Class Probabilities")

            for class_name, probability in zip(
                class_names,
                prediction[0]
            ):
                probability_value = float(probability)

                st.write(
                    f"**{class_name.capitalize()}**: "
                    f"{probability_value * 100:.2f}%"
                )

                st.progress(
                    min(max(probability_value, 0.0), 1.0)
                )

    except Exception as e:
        st.error(
            f"MRI processing error: {type(e).__name__}: {e}"
        )


# Medical disclaimer
st.markdown("""
---

**Medical Disclaimer**

This application is for educational and project demonstration purposes only.
AI predictions and generated explanations are not medical diagnoses.
Please consult a qualified medical professional for interpretation of MRI scans.
""")

st.caption(
    "AI Medical Assistant | Brain MRI Classification"
)

st.caption(
    "Built using ResNet50 + TensorFlow + Streamlit + Gemini"
)
