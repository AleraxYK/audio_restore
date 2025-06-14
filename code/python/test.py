import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

from torch import hann_window
from torchaudio.transforms import Spectrogram, GriffinLim, Resample, AmplitudeToDB
from torchaudio.functional import lowpass_biquad, highpass_biquad

# ==========================
# Configurazione
# ==========================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Percorsi
MODEL_PATH   = "model/UNet_audio_restoration.pth"
INPUT_FILE   = "output_musicgen_8.wav"              # file degradato
RESTORED_TMP = "restored_tmp.wav"        # output intermedio da U‐Net+GL
OUTPUT_FILE  = "restored_crossover.wav"  # output finale mixato

# Parametri campionamento & STFT
MODEL_SR     = 16000
OUTPUT_SR    = 44100
N_FFT        = 1024
HOP_LENGTH   = 16
WIN_LENGTH   = N_FFT
N_ITER_GL    = 64
CUTOFF        = 40   # Hz per filtro crossover
ALPHA         = 0.5    # peso per il ramo high-pass

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
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = UNetBlock(1,  64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128,256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(256,512)
        self.up3  = nn.ConvTranspose2d(512,256,2,2); self.dec3 = UNetBlock(512,256)
        self.up2  = nn.ConvTranspose2d(256,128,2,2); self.dec2 = UNetBlock(256,128)
        self.up1  = nn.ConvTranspose2d(128,64,2,2);  self.dec1 = UNetBlock(128,64)
        self.final_conv = nn.Conv2d(64,1,1)

    def center_crop(self, src, tgt):
        _,_,h,w   = src.shape
        _,_,th,tw = tgt.shape
        dh, dw = (h-th)//2, (w-tw)//2
        return src[:,:,dh:dh+th, dw:dw+tw]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.up3(b);  e3c = self.center_crop(e3, d3);  d3 = self.dec3(torch.cat([d3, e3c], 1))
        d2 = self.up2(d3); e2c = self.center_crop(e2, d2);  d2 = self.dec2(torch.cat([d2, e2c], 1))
        d1 = self.up1(d2); e1c = self.center_crop(e1, d1);  d1 = self.dec1(torch.cat([d1, e1c], 1))
        return self.final_conv(d1)


model = UNet().to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
state = ckpt.get('model_state', ckpt)
model.load_state_dict(state, strict=False)
model.eval()


spec = Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, power=2.0).to(DEVICE)
gl   = GriffinLim(
    n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    window_fn=torch.hann_window, n_iter=N_ITER_GL
).to(DEVICE)
to_db = AmplitudeToDB(stype='power')


wav, sr = torchaudio.load(INPUT_FILE)
wav = wav.float()
if sr != MODEL_SR:
    wav = Resample(sr, MODEL_SR)(wav)
wav = wav.mean(0, keepdim=True).to(DEVICE)

# power-spectrogram
P    = spec(wav)            
inp  = P.unsqueeze(1)       
with torch.no_grad():
    pred = model(inp).squeeze(1)  

# align frequency bins
TARGET_BINS = N_FFT//2 + 1
Fp = pred.size(1)
if Fp < TARGET_BINS:
    pred = F.pad(pred, (0,0, 0, TARGET_BINS - Fp))
elif Fp > TARGET_BINS:
    pred = pred[:, :TARGET_BINS, :]

mag = torch.sqrt(torch.clamp(pred, min=0.0))  # [1,513,T]
restored = gl(mag).cpu()                      # [1, L]
torchaudio.save(RESTORED_TMP, restored, MODEL_SR)

# upsample to OUTPUT_SR
if MODEL_SR != OUTPUT_SR:
    restored = Resample(MODEL_SR, OUTPUT_SR)(restored)

# ==========================
# 2) Time-domain crossover mix (4th-order via cascaded biquads)
# ==========================
# load original at OUTPUT_SR
orig, sr_o = torchaudio.load(INPUT_FILE)
if sr_o != OUTPUT_SR:
    orig = Resample(sr_o, OUTPUT_SR)(orig)
orig = orig.mean(0, keepdim=True).to(DEVICE)
rest = restored.to(DEVICE)

# cascaded low-pass on restored (4th order)
low = lowpass_biquad(rest, OUTPUT_SR, CUTOFF)
low = lowpass_biquad(low,       OUTPUT_SR, CUTOFF)

# cascaded high-pass on original (4th order)
high = highpass_biquad(orig, OUTPUT_SR, CUTOFF)
high = highpass_biquad(high,    OUTPUT_SR, CUTOFF)

# align time-length
L_low  = low.size(-1)
L_high = high.size(-1)
L      = min(L_low, L_high)
low    = low[..., :L]
high   = high[..., :L]

# mix with reduced high-pass weight
mixed = low + ALPHA * high
mixed = mixed / mixed.abs().max() * 0.95

torchaudio.save(OUTPUT_FILE, mixed.cpu(), OUTPUT_SR)
print(f"✅ Crossover mix saved to {OUTPUT_FILE}")

# ==========================
# 3) Debug: save spectrograms
# ==========================
os.makedirs('spec_debug', exist_ok=True)
for name, wv in [("degraded", orig), ("restored", rest), ("mixed", mixed)]:
    Pdb = to_db(spec(wv.cpu()))[0]
    plt.figure(figsize=(6,3))
    plt.imshow(Pdb.numpy(), origin='lower', aspect='auto',
               extent=[0, Pdb.shape[1]*HOP_LENGTH/OUTPUT_SR, 0, OUTPUT_SR/2])
    plt.title(f"{name.capitalize()} Spectrogram (dB)")
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.savefig(f"spec_debug/{name}.png")
    plt.close()
print("🔍 Spectrograms saved in spec_debug/")  