import torch
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import sounddevice as sd
import numpy as np
from noisereduce import reduce_noise
from scipy.io.wavfile import write
from io import BytesIO

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Initialize models and processors globally
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)

# Audio preprocessing to reduce noise
def preprocess_audio(audio):
    noise_sample = audio[:16000]  # Use the first second as noise sample
    return reduce_noise(audio_clip=np.array(audio), noise_clip=np.array(noise_sample))

# Record audio
def record_audio(duration=5, fs=16000):
    print("Start speaking...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")
    return audio.squeeze()

# Transcription function
def transcribe(audio):
    processed_audio = preprocess_audio(audio)
    input_features = whisper_processor(processed_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    generated_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.strip()

# Query response cleaning
def clean_response(response):
    return response.split(">>")[0].strip()

# Generate query response (placeholder for actual interaction with an LLM)
def query(input_text):
    # Mock response for demonstration
    return f"The transcription was: {input_text}"

# Text-to-speech synthesis
def synthesise(text):
    inputs = tts_processor(text=text, return_tensors="pt")
    speech = tts_model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings=None)
    return speech.cpu().numpy()

# Play generated audio
def play_audio(audio, fs=16000):
    sd.play(audio, samplerate=fs)
    sd.wait()

# Main function
if __name__ == "__main__":
    try:
        print("Listening for wake word...")
        while True:
            # Simulate wake word detection (you can replace this with your actual wake word logic)
            wake_word_detected = input("Type 'wake' to simulate wake word detection: ").strip().lower() == "wake"
            if wake_word_detected:
                print("Wake word detected!")
                
                # Record and process audio
                recorded_audio = record_audio()
                transcription = transcribe(recorded_audio)
                print(f"Transcription: {transcription}")
                
                # Query response
                response = query(transcription)
                response = clean_response(response)
                print(f"Response: {response}")
                
                # Synthesize and play audio response
                generated_audio = synthesise(response)
                print("Playing response...")
                play_audio(generated_audio)

    except Exception as e:
        print(f"An error occurred: {e}")
