# Load model and processor directly from Hugging Face
from transformers import AutoProcessor, AutoModelForAudioClassification
import torch
import torch.nn.functional as F
import librosa

processor = AutoProcessor.from_pretrained("Tagoreparaizo/IAUnit")
model = AutoModelForAudioClassification.from_pretrained("Tagoreparaizo/IAUnit")

file_path = "sad.wav" #Caminho do áudio
audio, sample_rate = librosa.load(file_path, sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(**inputs).logits

probs = F.softmax(logits, dim=-1)

predicted_id = torch.argmax(probs, dim=-1).item()
confidence = probs[0, predicted_id].item()

emotion_labels = ["angry", "disgusted", "happy", "fearful", "neutral", "sad", "surprised"]
predicted_emotion = emotion_labels[predicted_id]

print(f"Predicted Emotion: {predicted_emotion}")
print(f"predicted ID: {predicted_id}")
print(f"Confidence: {confidence:.2f}")
