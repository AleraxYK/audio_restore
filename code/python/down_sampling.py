import torchaudio
import torchaudio.transforms as T

# File input ad alta qualità
input_path = "test.wav"
output_path = "test_degraded.wav"

# Parametri
original_sr = 44100
target_sr = 8000

# Carica audio
waveform, sr = torchaudio.load(input_path)
if sr != original_sr:
    print(f"⚠️ Warning: sample rate del file è {sr}, atteso {original_sr}")

# Conversione mono
waveform = waveform.mean(dim=0, keepdim=True)

# Downsampling
resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
degraded = resampler(waveform)

# Upsampling di nuovo a 16k (come nel training)
final_resample = T.Resample(orig_freq=target_sr, new_freq=8000)
restored_rate_audio = final_resample(degraded)

# Salva
torchaudio.save(output_path, restored_rate_audio, 8000)
print(f"✅ File degradato salvato come '{output_path}'")