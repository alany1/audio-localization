import glob

import librosa
import numpy as np
import torch
from params_proto import Proto, ParamsProto, PrefixProto

from scripts.AudioCLIP.utils.transforms import ToTensor1D


class AudioArgs(PrefixProto):
    path = "../examples/ocean-wave-1.wav"
    device = "cuda"
    sample_rate = 44100
    # sample_rate = 22000

def get_audio_embeddings(model):
    """
    Get the audio embeddings for the audio file at `audio_path` using the given model.
    """
    print("Processing audio", AudioArgs.path)
    model.to(AudioArgs.device)
    model.eval()

    audio_transforms = ToTensor1D()

    track, _ = librosa.load(AudioArgs.path, sr=AudioArgs.sample_rate, dtype=np.float32)
    audio = audio_transforms(track.reshape(1, -1)).to(AudioArgs.device)
    ((audio_features, _, _), _), _ = model(audio=audio)
    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)

    return audio_features

