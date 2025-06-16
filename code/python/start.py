import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample, Spectrogram, GriffinLim, AmplitudeToDB
import matplotlib.pyplot as plt


DEVICE      = 'cpu'
MODEL_PATH  = 'models/UNet_audio_restoration.pth'
INPUT_FILE  = 'data/music_gen/output_musicgen_8.wav'
TEMP_FILE   = 'restored_output.wav'
DEBUG_DIR   = 'spec_debug'
OUTPUT_FIN  = 'generated_audio/audio_8_restored.wav'


MODEL_SR    = 16000
OUTPUT_SR   = 44100
N_FFT       = 4096
HOP_LENGTH  = 128
WIN_LENGTH  = N_FFT
TARGET_BINS = N_FFT // 2 + 1
HIGH_WEIGHT = 3.0
W_RESTORED  = 0.3
W_DEGRADED  = 1.0 - W_RESTORED

os.makedirs(DEBUG_DIR, exist_ok=True)

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
        self.enc1 = UNetBlock(1,64)
        self.enc2 = UNetBlock(64,128)
        self.enc3 = UNetBlock(128,256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(256,512)
        self.up3 = nn.ConvTranspose2d(512,256,2,2); self.dec3 = UNetBlock(512,256)
        self.up2 = nn.ConvTranspose2d(256,128,2,2); self.dec2 = UNetBlock(256,128)
        self.up1 = nn.ConvTranspose2d(128, 64,2,2);  self.dec1 = UNetBlock(128, 64)
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
        d3 = self.up3(b);  e3c = self.center_crop(e3, d3); d3 = self.dec3(torch.cat([d3,e3c],1))
        d2 = self.up2(d3); e2c = self.center_crop(e2, d2); d2 = self.dec2(torch.cat([d2,e2c],1))
        d1 = self.up1(d2); e1c = self.center_crop(e1, d1); d1 = self.dec1(torch.cat([d1,e1c],1))
        return self.final_conv(d1)

model = UNet().to(DEVICE)
sd = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(sd, strict=False)
model.eval()

spec = Spectrogram(
    n_fft=N_FFT, hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH, power=2.0,
    center=False
).to(DEVICE)

gl = GriffinLim(
    n_fft=N_FFT, hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH, power=1.0,
    n_iter=64
).to(DEVICE)

to_db = AmplitudeToDB(stype='power')

wav, sr = torchaudio.load(INPUT_FILE)
wav = wav.mean(0, keepdim=True)
orig_len = wav.size(-1)
if sr != MODEL_SR:
    wav = Resample(sr, MODEL_SR)(wav)
wav = wav.to(DEVICE).float()
P     = spec(wav)
P_db  = to_db(P)
mean_db, std_db = P_db.mean(), P_db.std()
Pn    = (P_db - mean_db) / (std_db + 1e-6)

with torch.no_grad():
    delta = model(Pn.unsqueeze(1)).squeeze(1)

b, f_pred, t_pred = delta.shape
if f_pred < TARGET_BINS:
    delta = F.pad(delta, (0,0, 0, TARGET_BINS-f_pred))
elif f_pred > TARGET_BINS:
    delta = delta[:, :TARGET_BINS, :]

_, _, T = P_db.shape
if t_pred < T:
    delta = F.pad(delta, (0, T-t_pred))
elif t_pred > T:
    delta = delta[:, :, :T]

w = torch.linspace(1.0, HIGH_WEIGHT, TARGET_BINS, device=DEVICE)
delta = delta * w.view(1, TARGET_BINS, 1)

P_db_est = P_db + delta

P_est = 10**(P_db_est / 10.0)
mag   = torch.sqrt(P_est)

wav_rec = gl(mag)
wav_rec = wav_rec[:, :orig_len]

if MODEL_SR != OUTPUT_SR:
    wav_rec = Resample(MODEL_SR, OUTPUT_SR)(wav_rec)

torchaudio.save(TEMP_FILE, wav_rec.cpu(), OUTPUT_SR)

deg, sr_deg = torchaudio.load(INPUT_FILE)  
res, sr_res = torchaudio.load(TEMP_FILE)  

if deg.size(0) > 1:
    deg = deg.mean(0, keepdim=True)
if res.size(0) > 1:
    res = res.mean(0, keepdim=True)

if sr_deg != OUTPUT_SR:
    deg = Resample(sr_deg, OUTPUT_SR)(deg)
if sr_res != OUTPUT_SR:
    res = Resample(sr_res, OUTPUT_SR)(res)

L_deg = deg.size(1)
L_res = res.size(1)

if L_res > L_deg:
    start_trim = (L_res - L_deg) // 2
    res = res[:, start_trim:start_trim + L_deg]
    L_res = L_deg

start = max((L_deg - L_res) // 2, 0)
end   = start + L_res

deg_window   = deg[:, start:end]
mixed_window = W_DEGRADED * deg_window + W_RESTORED * res

out = deg.clone()
out[:, start:end] = mixed_window

peak = out.abs().max()
if peak > 1.0:
    out = out / peak * 0.95

torchaudio.save(OUTPUT_FIN, out, OUTPUT_SR)
print(f"File saved in: {OUTPUT_FIN}")

spec = Spectrogram(n_fft=1024, hop_length=32, power=2.0)
to_db = AmplitudeToDB(stype='power')

P   = spec(out)    
Pdb = to_db(P)[0]

plt.figure(figsize=(10, 4))
plt.imshow(
    Pdb.numpy(),
    origin='lower',
    aspect='auto',
    extent=[0, Pdb.shape[1]*32/OUTPUT_SR, 0, OUTPUT_SR/2]
)

plt.colorbar(label='dB')
plt.title("Final Mixed Spectrogram")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.tight_layout()

out_png = os.path.join(DEBUG_DIR, "final_spectrogram.png")
plt.savefig(out_png, dpi=500)
plt.close()

print(f"Spectrogram saved in {out_png}")

os.remove(TEMP_FILE)
