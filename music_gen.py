from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torchaudio
import torch
import pandas as pd
import os

# Percorso al file CSV
csv_file = "text_to_music_eval_prompts.csv"

# Directory output
output_dir = "generated_audio"
os.makedirs(output_dir, exist_ok=True)

# Carica modello e processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Carica CSV
df = pd.read_csv(csv_file)

for idx, row in df.iterrows():
    text_prompt = row['TextPrompt']
    print(f"Generating audio for prompt {idx}: {text_prompt}")

    # Prepara input e sposta su device
    inputs = processor(text=[text_prompt], return_tensors="pt").to(device)

    # Genera audio (circa 16 secondi a 32kHz)
    audio_values = model.generate(**inputs, max_new_tokens=1024)

    # Estrai audio e sistema dimensioni
    waveform = audio_values[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # aggiungi dimensione canale

    # Salva file wav
    output_filename = os.path.join(output_dir, f"output_musicgen_{idx}.wav")
    torchaudio.save(output_filename, waveform.cpu(), sample_rate=32000)

    print(f"Saved {output_filename}")
