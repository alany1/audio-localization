import wav2clip
from params_proto import Proto, ParamsProto
import torch, torchaudio
import numpy as np

class AudioArgs(ParamsProto):
    source_path: str = Proto(help='Path to the audio file.')
    # model_str = Proto(type=str, default='', help='Model to use for embeddings.')
def get_embeddings(audio_path):
    """Returns the embeddings of the audio file at the given path."""
    print("Getting embeddings for audio file at path: {}".format(AudioArgs.source_path))
    wave_form, sample_rate = torchaudio.load(AudioArgs.source_path)
    wave_form = torch.mean(wave_form, dim=0)
    model = wav2clip.get_model()
    print("Received Model")
    return wav2clip.embed_audio(np.array(wave_form), model)

if __name__ == '__main__':
    AudioArgs.source_path = 'examples/ocean-wave-1.wav'
    print(get_embeddings(AudioArgs.source_path).shape)