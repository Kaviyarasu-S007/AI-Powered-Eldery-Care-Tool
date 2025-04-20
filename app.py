# import streamlit as st
# from gradio_client import Client, handle_file
# import tempfile

# # Initialize the Gradio client
# client = Client("nvidia/audio-flamingo-2")

# st.set_page_config(page_title="Audio Analyzer", layout="centered")
# st.title("🔊 AI Powered Eldery Care Tool")
# st.markdown("Upload an audio file and ask a question about it (e.g., emotion, content, etc.)")

# # --- Upload audio ---
# uploaded_file = st.file_uploader("📁 Upload your audio file", type=["wav", "mp3", "m4a"])

# # --- Ask a question ---
# question = st.text_input("✍️ Enter your question about the audio", value="Describe the emotion of speaker in audio.")

# # --- Show audio player ---
# if uploaded_file:
#     st.audio(uploaded_file, format="audio/wav")

# # --- Analyze button ---
# if st.button("🔍 Analyze"):
#     if not uploaded_file or not question.strip():
#         st.warning("Please upload an audio file and enter a question.")
#     else:
#         # Save file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             tmp_path = tmp_file.name

#         with st.spinner("Analyzing with NVIDIA AudioFlamingo..."):
#             try:
#                 result = client.predict(
#                     filepath=handle_file(tmp_path),
#                     question=question.strip(),
#                     api_name="/predict"
#                 )
#                 st.success("Analysis complete!")
#                 st.write("### 🧠 Model Response:")
#                 st.write(result)
#             except Exception as e:
#                 st.error(f"Error during prediction: {e}")


########################### 2 MODELS ################################################

# import streamlit as st
# from gradio_client import Client, handle_file
# import tempfile
# import os

# # Title & Sidebar setup
# st.set_page_config(page_title="Audio Analyzer Dashboard", layout="centered")
# st.sidebar.title("🧠 Select Model")
# model_choice = st.sidebar.radio("Choose a model", ["🔊 AudioFlamingo (NVIDIA)", "🗣️ SpeechLLM (Skit AI)"])

# st.title("🔊 AI Powered Elderly Care Tool")

# # ---------------------------------------
# # MODEL 1: NVIDIA AudioFlamingo
# # ---------------------------------------
# if model_choice == "🔊 AudioFlamingo (NVIDIA)":
#     st.subheader("🎧 Ask questions about audio (Emotion, Content, etc.)")

#     # Initialize Gradio client
#     flamingo_client = Client("nvidia/audio-flamingo-2")

#     # Upload audio
#     uploaded_file = st.file_uploader("📁 Upload your audio file", type=["wav", "mp3", "m4a"])
#     question = st.text_input("✍️ Enter your question", value="Describe the emotion of speaker in audio.")

#     # Show audio player
#     if uploaded_file:
#         st.audio(uploaded_file, format="audio/wav")

#     if st.button("🔍 Analyze"):
#         if not uploaded_file or not question.strip():
#             st.warning("Please upload an audio file and enter a question.")
#         else:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#                 tmp_file.write(uploaded_file.read())
#                 tmp_path = tmp_file.name

#             with st.spinner("Analyzing with NVIDIA AudioFlamingo..."):
#                 try:
#                     result = flamingo_client.predict(
#                         filepath=handle_file(tmp_path),
#                         question=question.strip(),
#                         api_name="/predict"
#                     )
#                     st.success("✅ Analysis complete!")
#                     st.markdown("### 🧠 Model Response:")
#                     st.write(result)
#                 except Exception as e:
#                     st.error(f"❌ Error: {e}")
#             os.remove(tmp_path)


# # ---------------------------------------
# # MODEL 2: SpeechLLM (Skit AI)
# # ---------------------------------------
# elif model_choice == "🗣️ SpeechLLM (Skit AI)":
#     import torchaudio
#     import torch
#     from transformers import AutoModel

#     st.subheader("🎤 Ask instructions to SpeechLLM (e.g., number of speakers)")

#     uploaded_file = st.file_uploader("📁 Upload audio (WAV recommended)", type=["wav"])
#     instruction = st.text_input("✍️ Instruction", value="How many speakers are there in the audio?")

#     if uploaded_file:
#         st.audio(uploaded_file, format="audio/wav")

#     if st.button("🚀 Run SpeechLLM"):
#         if not uploaded_file or not instruction.strip():
#             st.warning("Please upload a WAV file and enter an instruction.")
#         else:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#                 tmp_file.write(uploaded_file.read())
#                 audio_path = tmp_file.name

#             with st.spinner("Running SpeechLLM model..."):
#                 try:
#                     model = AutoModel.from_pretrained("skit-ai/speechllm-2B", trust_remote_code=True)
#                     model = model.to(dtype=torch.float32)
#                     waveform, sample_rate = torchaudio.load(audio_path)
#                     waveform = waveform.to(torch.float32)

#                     output = model.generate_meta(
#                         audio_path=audio_path,
#                         instruction=instruction,
#                         max_new_tokens=500
#                     )
#                     st.success("✅ SpeechLLM finished!")
#                     st.markdown("### 🧠 Model Response:")
#                     st.write(output)
#                 except Exception as e:
#                     st.error(f"❌ Error: {e}")
#             os.remove(audio_path)


import streamlit as st
from gradio_client import Client, handle_file
import tempfile
import os

st.set_page_config(page_title="Audio Analyzer Dashboard", layout="centered")
st.sidebar.title("🧠 Select Model")
model_choice = st.sidebar.radio("Choose a model", ["🔊 AudioFlamingo (NVIDIA)", "🗣️ SpeechLLM (Skit AI)"])

st.title("🔊 AI Powered Elderly Care Tool")

# ----------------------
# MODEL 1: AudioFlamingo
# ----------------------
@st.cache_resource
def load_audioflamingo_client():
    return Client("nvidia/audio-flamingo-2")

if model_choice == "🔊 AudioFlamingo (NVIDIA)":
    st.subheader("🎧 Ask questions about audio (Emotion, Content, etc.)")

    flamingo_client = load_audioflamingo_client()

    uploaded_file = st.file_uploader("📁 Upload your audio file", type=["wav", "mp3", "m4a"])
    question = st.text_input("✍️ Enter your question", value="Describe the emotion of speaker in audio.")

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍 Analyze"):
        if not uploaded_file or not question.strip():
            st.warning("Please upload an audio file and enter a question.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            with st.spinner("Analyzing with NVIDIA AudioFlamingo..."):
                try:
                    result = flamingo_client.predict(
                        filepath=handle_file(tmp_path),
                        question=question.strip(),
                        api_name="/predict"
                    )
                    st.success("✅ Analysis complete!")
                    st.markdown("### 🧠 Model Response:")
                    st.write(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            os.remove(tmp_path)

# ------------------------
# MODEL 2: SpeechLLM Skit
# ------------------------
elif model_choice == "🗣️ SpeechLLM (Skit AI)":
    import torchaudio
    import torch
    from transformers import AutoModel

    st.subheader("🎤 Ask instructions to SpeechLLM (e.g., number of speakers)")

    @st.cache_resource
    def load_speechllm_model():
        model = AutoModel.from_pretrained("skit-ai/speechllm-2B", trust_remote_code=True)
        return model.to(dtype=torch.float32)

    speechllm_model = load_speechllm_model()

    uploaded_file = st.file_uploader("📁 Upload audio (WAV recommended)", type=["wav"])
    instruction = st.text_input("✍️ Instruction", value="How many speakers are there in the audio?")

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

    if st.button("🚀 Run SpeechLLM"):
        if not uploaded_file or not instruction.strip():
            st.warning("Please upload a WAV file and enter an instruction.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name

            with st.spinner("Running SpeechLLM model..."):
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    waveform = waveform.to(torch.float32)

                    output = speechllm_model.generate_meta(
                        audio_path=audio_path,
                        instruction=instruction,
                        max_new_tokens=500
                    )
                    st.success("✅ SpeechLLM finished!")
                    st.markdown("### 🧠 Model Response:")
                    st.write(output)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            os.remove(audio_path)
