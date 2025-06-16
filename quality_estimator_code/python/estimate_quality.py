import torch
import torch.nn as nn
import torchaudio
import numpy as np
from scipy.signal import stft
from restore_code.python.start import INPUT_FILE, OUTPUT_FIN


mel_transform_base = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=96
    ),
    torchaudio.transforms.AmplitudeToDB()
)

class AudioQualityCNN(nn.Module):
    def __init__(self, n_mels=64):
        super(AudioQualityCNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),             

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),             

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))                  
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),                     
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()                      
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.regressor(x)
        return x.squeeze(1)  
    

def evaluate_audio_quality(model, original_audio, generated_audio, sample_rate=16000, duration=3.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    waveform, sr = torchaudio.load(original_audio)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    num_samples = int(sample_rate * duration)
    if waveform.shape[1] < num_samples:
        padding = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :num_samples]

    waveform = waveform.unsqueeze(0).to(device)

    spectrogram = mel_transform_base(waveform)

    with torch.no_grad():
        score = model(spectrogram)

    generated_waveform, sr_gen = torchaudio.load(generated_audio)
    if sr_gen != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr_gen, new_freq=sample_rate)
        generated_waveform = resampler(generated_waveform)
    if generated_waveform.shape[0] > 1:
        generated_waveform = generated_waveform.mean(dim=0, keepdim=True)
    if generated_waveform.shape[1] < num_samples:
        padding = num_samples - generated_waveform.shape[1]
        generated_waveform = torch.nn.functional.pad(generated_waveform, (0, padding))
    else:
        generated_waveform = generated_waveform[:, :num_samples]
    generated_waveform = generated_waveform.unsqueeze(0).to(device)
    generated_spectrogram = mel_transform_base(generated_waveform)
    with torch.no_grad():
        generated_score = model(generated_spectrogram)

    waveform = waveform.squeeze(0)
    generated_waveform = generated_waveform.squeeze(0)

    snr = compute_snr(waveform, generated_waveform)
    
    lsd = compute_lsd(waveform.numpy(), generated_waveform.numpy())

    return score.item(), generated_score.item(), snr, lsd

def compute_snr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB"""
    reference = reference.flatten()
    estimate = estimate.flatten()
    min_len = min(reference.numel(), estimate.numel())
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    noise = estimate - reference
    snr = 10 * torch.log10(reference.pow(2).mean() / (noise.pow(2).mean() + 1e-8))
    return snr.item()

def compute_lsd(reference: torch.Tensor, estimate: torch.Tensor, n_fft=512, hop_length=256) -> float:
    """Compute Log-Spectral Distance (LSD)"""
    reference = reference
    estimate = estimate

    _, _, ref_stft = stft(reference, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, est_stft = stft(estimate, nperseg=n_fft, noverlap=n_fft - hop_length)

    ref_mag = np.abs(ref_stft)
    est_mag = np.abs(est_stft)

    epsilon = 1e-10
    log_ref = np.log10(ref_mag + epsilon)
    log_est = np.log10(est_mag + epsilon)

    lsd = np.sqrt(np.mean((log_ref - log_est) ** 2, axis=0))
    return np.mean(lsd)


model = AudioQualityCNN()
model.load_state_dict(torch.load("models/CNN_quality_estimator.pth"))
QSFG = []
QSFR = []
SNR  = []
LSD  = []

quality_score_original, quality_score_generated, snr, lsd = evaluate_audio_quality(model, INPUT_FILE, OUTPUT_FIN)
print(f"Quality score for {INPUT_FILE}: {quality_score_original:.4f}")
print(f"Quality score for {OUTPUT_FIN}: {quality_score_generated:.4f}")
print(f"SNR: {snr:.2f} dB")
print(f"LSD: {lsd:.4f} (lower is better)")
QSFG.append(quality_score_original)
QSFR.append(quality_score_generated)
SNR.append(snr)
LSD.append(lsd)