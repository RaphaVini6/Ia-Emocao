import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForAudioClassification
import wave
import io
import tempfile
import os

# Load model and processor
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("Tagoreparaizo/IAUnit")
    model = AutoModelForAudioClassification.from_pretrained("Tagoreparaizo/IAUnit")
    return processor, model

def record_audio(duration, samplerate=16000):
    st.write("🎤 Gravando...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

def save_audio(audio_data, samplerate=16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        with wave.open(tmpfile.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        return tmpfile.name

def process_audio(audio_path, processor, model):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = F.softmax(logits, dim=-1)
    predicted_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_id].item()
    
    emotion_labels = ["raiva", "nojo", "felicidade", "medo", "neutro", "tristeza", "surpreso"]
    return emotion_labels[predicted_id], confidence

def main():
    st.title("🎭 Detecção de emoção")
    processor, model = load_model()
    
    tab1, tab2 = st.tabs(["Gravar Audio", "Upload de Audio"])
    
    with tab1:
        st.write("Grave um áudio")
        duration = st.slider("Duração da gravação (segundos)", 1, 10, 5)
        if st.button("🎤 Começar a gravar"):
            audio_data = record_audio(duration)
            audio_file = save_audio(audio_data)
            
            with st.spinner("Analisando emoção..."):
                emotion, confidence = process_audio(audio_file, processor, model)
                
            st.success(f"Emoção detectada: **{emotion}**")
            st.progress(confidence)
            st.write(f"Confidence: {confidence:.2%}")
            
            os.unlink(audio_file)  # Clean up temporary file
    
    with tab2:
        st.write("Faça upload de um arquivo de áudio (WAV)")
        uploaded_file = st.file_uploader("Choose a file", type=['wav'])
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                
            with st.spinner("Analisando Emoção..."):
                emotion, confidence = process_audio(tmpfile.name, processor, model)
            
            st.success(f"Emoção Detectada: **{emotion}**")
            st.progress(confidence)
            st.write(f"Confidence: {confidence:.2%}")
            
            os.unlink(tmpfile.name)  # Clean up temporary file

if __name__ == "__main__":
    main()