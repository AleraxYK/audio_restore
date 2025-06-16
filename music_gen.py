from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torchaudio
import torch
import pandas as pd
import os

csv_file = "text_to_music_eval_prompts.csv"

output_dir = "generated_audio"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

df = pd.read_csv(csv_file)

for idx, row in df.iterrows():
    text_prompt = row['TextPrompt']
    print(f"Generating audio for prompt {idx}: {text_prompt}")

    inputs = processor(text=[text_prompt], return_tensors="pt").to(device)

    audio_values = model.generate(**inputs, max_new_tokens=1024)

    waveform = audio_values[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0) 

    output_filename = os.path.join(output_dir, f"output_musicgen_{idx}.wav")
    torchaudio.save(output_filename, waveform.cpu(), sample_rate=32000)

    print(f"Saved {output_filename}")
