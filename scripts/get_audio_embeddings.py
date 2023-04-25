import wav2clip
from params_proto import Proto, ParamsProto
import torch, torchaudio
import numpy as np

class AudioArgs(ParamsProto):
    audio_path: str = Proto(help='Path to the audio file.')
    # model_str = Proto(type=str, default='', help='Model to use for embeddings.')

def get_audio_embeddings(audio_path):
    """
    Returns the embeddings of the audio file at the given path.
    """
    print("Getting embeddings for audio file at path: {}".format(audio_path))
    wave_form, sample_rate = torchaudio.load(audio_path)
    wave_form = torch.mean(wave_form, dim=0)
    model = wav2clip.get_model()
    print("Received Model")
    return wav2clip.embed_audio(np.array(wave_form), model)

if __name__ == '__main__':
    # AudioArgs.audio_path = 'examples/ocean-wave-1.wav'
    AudioArgs.audio_path = 'examples/car-ignition.wav'
    print(get_audio_embeddings(AudioArgs.audio_path).shape)
