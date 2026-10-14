
import time
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
    except Exception:
        return "Gemini API key missing. Check Streamlit Secrets."

    prompt = f"""
You are an educational medical information assistant.

An image classification model predicted: {predicted_class}.
Model confidence score: {confidence:.2f}%.

Explain this class in simple English for educational purposes.
Clarify that the confidence score is not a medical probability
and does not confirm a diagnosis. Recommend review by a
qualified radiologist. Do not infer findings beyond the prediction.
Keep the explanation concise.
"""

    client = genai.Client(api_key=api_key)

    # Retry transient server errors up to 3 attempts
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            if response.text:
                return response.text

            return "Gemini returned an empty response. Please retry."

        except Exception as e:
            error_text = str(e)
            is_temporary = (
                "503" in error_text
                or "429" in error_text
                or "UNAVAILABLE" in error_text.upper()
                or "RESOURCE_EXHAUSTED" in error_text.upper()
            )

            if is_temporary and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue

            return (
                f"GenAI explanation unavailable: "
                f"{type(e).__name__}: {error_text}"
            )


# Load trained MRI classification model
@st.cache_resource
def load_ai_model():
    return load_model("resnet50_brain_mri.keras")


try:
    model = load_ai_model()
except Exception as e:
    st.error(
        f"Could not load MRI model: {type(e).__name__}: {e}"
    )
    st.stop()


class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# Upload MRI image
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"],
    key="mri_uploader"
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()

        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")

        st.image(
            image.copy(),
            caption="Uploaded Brain MRI",
            width=400
        )

        if st.button("🔍 Analyze MRI"):
            with st.spinner("Analyzing MRI image..."):
                img = image.resize((224, 224))

                img_array = np.asarray(
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

                st.session_state["mri_result"] = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": prediction[0].tolist()
                }

                # Clear any previous explanation
                st.session_state.pop("mri_explanation", None)

        # Show saved prediction after Streamlit reruns
        result = st.session_state.get("mri_result")

        if result is not None:
            st.success(
                "Predicted Class: "
                + result["predicted_class"].capitalize()
            )

            st.metric(
                "Model Confidence",
                f'{result["confidence"]:.2f}%'
            )

            st.subheader("📊 Class Probabilities")

            for class_name, probability in zip(
                class_names,
                result["probabilities"]
            ):
                probability_value = float(probability)

                st.write(
                    f"**{class_name.capitalize()}**: "
                    f"{probability_value * 100:.2f}%"
                )

                st.progress(
                    min(max(probability_value, 0.0), 1.0)
                )

            st.subheader("✨ GenAI Explanation")

            if st.button("💬 Generate / Retry Explanation"):
                with st.spinner(
                    "Contacting Gemini. Please wait..."
                ):
                    explanation = generate_ai_explanation(
                        result["predicted_class"],
                        result["confidence"]
                    )

                st.session_state["mri_explanation"] = explanation

            explanation = st.session_state.get("mri_explanation")

            if explanation:
                if explanation.startswith(
                    "GenAI explanation unavailable:"
                ):
                    st.warning(explanation)
                else:
                    st.write(explanation)

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
The model may make mistakes. Consult a qualified medical professional
or radiologist for interpretation of MRI scans.
""")

st.caption(
    "AI Medical Assistant | Brain MRI Classification"
)

st.caption(
    "Built using ResNet50 + TensorFlow + Streamlit + Gemini"
)
