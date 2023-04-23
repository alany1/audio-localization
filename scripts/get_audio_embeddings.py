import wav2clip
import torch, torchaudio
import numpy as np
import clip
from utils import *
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt

def get_audio_embeddings(audio_path):
    """
    Returns the embeddings of the audio file at the given path.
    """
    print("Getting embeddings for audio file at path: {}".format(audio_path))
    wave_form, sample_rate = torchaudio.load(audio_path)
    wave_form = torch.mean(wave_form, dim=0)
    model = wav2clip.get_model()
    print("Received Model")
    return torch.flatten(torch.tensor(wav2clip.embed_audio(np.array(wave_form), model)))


if __name__ == '__main__':
    audio_file = FileArgs()
    audio_file.source_path = 'scripts/examples/ocean-wave-1.wav'
    audio_clip = get_audio_embeddings(audio_file.source_path)

    
