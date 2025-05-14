import streamlit as st
from gradio_client import Client, handle_file
import tempfile

# Initialize the Gradio client
client = Client("nvidia/audio-flamingo-2-0.5B")

st.set_page_config(page_title="Audio Analyzer", layout="centered")
st.title("🔊 AI Powered Eldery Care Tool")
st.markdown("Upload an audio file and ask a question about it (e.g., emotion, content, etc.)")

# --- Upload audio ---
uploaded_file = st.file_uploader("📁 Upload your audio file", type=["wav", "mp3", "m4a"])

# --- Ask a question ---
question = st.text_input("✍️ Enter your question about the audio", value="Describe the emotion of speaker in audio.")

# --- Show audio player ---
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

# --- Analyze button ---
if st.button("🔍 Analyze"):
    if not uploaded_file or not question.strip():
        st.warning("Please upload an audio file and enter a question.")
    else:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("Analyzing with NVIDIA AudioFlamingo..."):
            try:
                result = client.predict(
                    filepath=handle_file(tmp_path),
                    question=question.strip(),
                    api_name="/predict"
                )
                st.success("Analysis complete!")
                st.write("### 🧠 Model Response:")
                st.write(result)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
