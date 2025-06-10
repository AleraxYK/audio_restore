import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import hann_window
from torchaudio.transforms import Resample, Spectrogram, GriffinLim

# ==========================
# Configurazione
# ==========================
DEVICE      = torch.device("cpu")
MODEL_PATH  = "model/UNet_audio_restoration.pth"           # checkpoint con solo model.state_dict()
INPUT_FILE  = "000002_degraded.wav"       # file degradato in input
OUTPUT_FILE = "restored_44k1.wav"       # file restaurato in output
MODEL_SR    = 16000                      # sample rate per il modello
OUTPUT_SR   = 44100                      # sample rate finale desiderato
N_FFT       = 1024
HOP_LENGTH  = 32                        # 75% overlap
WIN_LENGTH  = N_FFT                      # finestra pari a N_FFT
TARGET_FREQ = N_FFT // 2 + 1             # 513 bins

# ==========================
# Definizione del modello UNet
# ==========================
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = UNetBlock(1,64)
        self.enc2 = UNetBlock(64,128)
        self.enc3 = UNetBlock(128,256)
        self.bottleneck = UNetBlock(256,512)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(512,256,2,2)
        self.dec3 = UNetBlock(512,256)
        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2 = UNetBlock(256,128)
        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1 = UNetBlock(128,64)
        self.final_conv = nn.Conv2d(64,1,1)

    def center_crop(self, src, tgt):
        _,_,h,w   = src.shape
        _,_,th,tw = tgt.shape
        dh, dw = (h-th)//2, (w-tw)//2
        return src[:,:,dh:dh+th, dw:dw+tw]

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.up3(b); e3c = self.center_crop(e3,d3)
        d3 = self.dec3(torch.cat([d3,e3c],1))
        d2 = self.up2(d3); e2c = self.center_crop(e2,d2)
        d2 = self.dec2(torch.cat([d2,e2c],1))
        d1 = self.up1(d2); e1c = self.center_crop(e1,d1)
        d1 = self.dec1(torch.cat([d1,e1c],1))
        return self.final_conv(d1)

# ==========================
# Caricamento del modello
# ==========================
model = UNet().to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
state = ckpt.get('model_state', ckpt)
model.load_state_dict(state)
model.eval()

# ==========================
# Trasformazioni con finestra di Hann (STFT manuale)
# ==========================
# Creazione finestra di Hann per STFT e ISTFT
window = hann_window(N_FFT, periodic=True).to(DEVICE)

# Rimuoviamo Spectrogram e useremo torch.stft direttamente

def power_spec(wav_tensor):
    # wav_tensor: [1, L]
    stft_complex = torch.stft(
        wav_tensor,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True
    )  # [1, freq, time]
    return stft_complex.abs()**2  # power spectrogram

# GriffinLim con finestra di Hann
try:
    gl = GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        power=1.0,
        n_iter=64
    ).to(DEVICE)
except TypeError:
    # Se la tua versione non supporta window, usa senza
    gl = GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        power=1.0,
        n_iter=64
    ).to(DEVICE)

# ==========================
wav, sr = torchaudio.load(INPUT_FILE)

# 1) Resample DOWN
if sr != MODEL_SR:
    wav = Resample(orig_freq=sr, new_freq=MODEL_SR)(wav)
    sr = MODEL_SR
    print(f"⚠︎ Resampled down to {MODEL_SR} Hz for model input")
# 2) Mono + DEVICE
wav = wav.mean(0, keepdim=True).to(DEVICE)

# 3) Power Spectrogram
P = power_spec(wav)  # [1, 513, T]

# 4) Add batch+channel dims
inp = P.unsqueeze(1)  # [1,1,513,T]

# 5) UNet inference
with torch.no_grad():
    pred_P = model(inp)  # [1,1,513,T]
pred_P = pred_P.squeeze(1)  # [1,513,T]

# 6) Ensure correct freq bins
batch, f, t = pred_P.shape
if f < TARGET_FREQ:
    pred_P = F.pad(pred_P, (0,0,0, TARGET_FREQ - f))
elif f > TARGET_FREQ:
    pred_P = pred_P[:, :TARGET_FREQ, :]

# 7) Power → magnitude
mag = torch.sqrt(torch.clamp(pred_P, min=0.0))  # [1,513,T]

# 8) Griffin-Lim reconstruction
time_wav = gl(mag)[0]  # [T]

# 9) Resample UP
if MODEL_SR != OUTPUT_SR:
    time_wav = Resample(orig_freq=MODEL_SR, new_freq=OUTPUT_SR)(time_wav.unsqueeze(0))
    print(f"⚠︎ Upsampled to {OUTPUT_SR} Hz for output")

# 10) Save final audio
torchaudio.save(OUTPUT_FILE, time_wav.cpu(), OUTPUT_SR)
print(f"✅ Audio restaurato salvato in: {OUTPUT_FILE}")

# 11) Salvataggio spettrogrammi per debug
import matplotlib.pyplot as plt
from torchaudio.transforms import Spectrogram, AmplitudeToDB

os.makedirs('spec_debug', exist_ok=True)

# 11.a) Spettrogramma del file degradato (a MODEL_SR)
wav_d, sr_d = torchaudio.load(INPUT_FILE)
if sr_d != MODEL_SR:
    wav_d = Resample(orig_freq=sr_d, new_freq=MODEL_SR)(wav_d)
wav_d = wav_d.mean(0, keepdim=True)
spec_cpu = Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0)
mag_d = spec_cpu(wav_d)
db_d = AmplitudeToDB(stype='power')(mag_d)[0]
plt.figure(figsize=(8,4))
plt.imshow(db_d.numpy(), origin='lower', aspect='auto', extent=[0, db_d.shape[1]*HOP_LENGTH/MODEL_SR, 0, MODEL_SR/2])
plt.colorbar(label='dB')
plt.title('Degraded Spectrogram (dB)')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.tight_layout()
plt.savefig('spec_debug/degraded_db_spec.png')
plt.close()

# 11.b) Spettrogramma del file restaurato (a OUTPUT_SR)
wav_r, sr_r = torchaudio.load(OUTPUT_FILE)
wav_r = wav_r.mean(0, keepdim=True)
spec_cpu = Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0)
mag_r = spec_cpu(wav_r)
db_r = AmplitudeToDB(stype='power')(mag_r)[0]
plt.figure(figsize=(8,4))
plt.imshow(db_r.numpy(), origin='lower', aspect='auto', extent=[0, db_r.shape[1]*HOP_LENGTH/OUTPUT_SR, 0, OUTPUT_SR/2])
plt.colorbar(label='dB')
plt.title('Restored Spectrogram (dB)')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.tight_layout()
plt.savefig('spec_debug/restored_db_spec.png')
plt.close()
print("🔍 Spettrogrammi salvati in spec_debug/")
