import torch
import torchaudio
from model.audioclip import AudioCLIP


# Load the pre-trained model
model = AudioCLIP(pretrained="AudioCLIP/assets/AudioCLIP-Full-Training.pt")
model.eval()
for module in model.modules():
    module.eval()

# Load an audio file
# audio_path = "examples/ocean-wave-1.wav"
# audio_path = "examples/car-ignition.wav"
# audio_path = "examples/dirt.mp3"
audio_path = "examples/chicken-1.wav"

waveform, sample_rate = torchaudio.load(audio_path)

# If the audio has multiple channels, average them to get a single-channel waveform
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.unsqueeze(0)
    # print(waveform.shape)
# Resample the audio wavefo rm to 44.1 kHz, as expected by the model

resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)
waveform = resampler(waveform)

# Get the audio embeddings
with torch.no_grad():
    audio_embeddings = model.encode_audio(waveform)

print("Audio embeddings shape:", audio_embeddings.shape)

# save the result
torch.save(audio_embeddings, "examples/chicken.pt")
print("Audio embeddings saved to examples/test_audio.pt")
