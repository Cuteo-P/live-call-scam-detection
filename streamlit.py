import streamlit as st
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from collections import deque
import torch
import time
from sbert_classifier import SBERTClassifier

# initalize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = WhisperModel("medium.en", device='cuda')
sbert_model = SBERTClassifier(sbert_model_name="sbert_model_v3", num_classes=2).to(device)
state_dict = torch.load("classifier_v3.pt", map_location=device)
sbert_model.classifier.load_state_dict(state_dict)
sbert_model.eval()

# Audio params
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
CONTEXT_SIZE = 2 # seconds

# State params
alpha = 0.3 # for EMA calculation a.k.a (Скользящая средняя)
overall_probs = None
overall_label = None
scam_counter = 0
context_buffer = deque(maxlen=CONTEXT_SIZE)
audio_buffer = []

st.title("Live Scam Call Detector")
st.write("Recording audio in chunks and updating predictions live.")

status_placeholder = st.empty()
overall_placeholder = st.empty()
transcript_placeholder = st.empty()

def record_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_buffer.extend(indata[:, 0])

def process_chunk(chunk_text, context_text):
    global overall_probs, overall_label, scam_counter

    # get probabilities
    with torch.no_grad():
        logits_context = sbert_model([context_text])
        probs_context = torch.softmax(logits_context, dim=1)[0].cpu().numpy()

    # get overall probability based on current + previous chunks using EMA
    if overall_probs is None:
        overall_probs = probs_context
    else:
        overall_probs = alpha * probs_context + (1 - alpha) * overall_probs

    overall_label = int(np.argmax(overall_probs))

    # Update scam counter
    scam_counter = scam_counter + 1 if overall_label == 1 else 0

    # status messages
    if scam_counter >= 5:
        color = "red"
        message = "🚨 Scammer detected! Call aborted."
    elif overall_label == 0:
        color = "green"
        message = "✅ Your call is safe"
    else:
        color = "orange"
        message = "⚠️ Possible scam"

    # Update UI
    status_placeholder.markdown(f"<b>Chunk:</b> {chunk_text}", unsafe_allow_html=True)
    overall_placeholder.markdown(
        f"<h3 style='color:{color}'>{message}</h3>",
        unsafe_allow_html=True
    )
    transcript_placeholder.text(context_text)

    return scam_counter >= 5

# Streamlit button to start recording
if st.button("Start Recording"):
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=record_callback):
        st.info("Recording... Press Stop to end.")
        stop_button = st.button("Stop Recording")
        try:
            while not stop_button:
                if len(audio_buffer) >= CHUNK_SIZE:
                    chunk = np.array(audio_buffer[:CHUNK_SIZE], dtype=np.float32)
                    del audio_buffer[:CHUNK_SIZE]
                    chunk = chunk / (np.max(np.abs(chunk)) + 1e-7)

                    # get transcribed text from audio input
                    segments, info = whisper_model.transcribe(chunk, beam_size=5)
                    chunk_text = " ".join([seg.text for seg in segments])
                    context_buffer.append(chunk_text)
                    context_text = " ".join(context_buffer)

                    # current chunk with context chunks are processed
                    if process_chunk(chunk_text, context_text):
                        break # breaks if scam_counter >= 5

                    time.sleep(0.1)

        except KeyboardInterrupt:
            st.warning("Recording stopped manually.")
