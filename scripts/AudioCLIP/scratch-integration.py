import os
import sys
import glob

import librosa
import librosa.display

import simplejpeg
import numpy as np

import torch
import torchvision as tv

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import io
from torchvision.transforms import ToTensor
from tqdm import tqdm

# from IPython.display import Audio, display

sys.path.append(os.path.abspath(f"{os.getcwd()}/.."))

from model import AudioCLIP
from utils.transforms import ToTensor1D


torch.set_grad_enabled(False)

MODEL_FILENAME = "AudioCLIP-Full-Training.pt"
# derived from ESResNeXt
SAMPLE_RATE = 44100
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

# LABELS = ['cat', 'thunderstorm', 'coughing', 'alarm clock', 'car horn']
# LABELS = ['ocean', 'water', 'beach', 'sand', 'thunderstorm', 'cat']
LABELS = ["car", "driving", "honk", "driving my new car", "whiteboard", "ocean"]
aclp = AudioCLIP(pretrained=f"assets/{MODEL_FILENAME}")
aclp.eval()

audio_transforms = ToTensor1D()

image_transforms = tv.transforms.Compose(
    [
        # tv.transforms.ToTensor(),
        tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        tv.transforms.CenterCrop(IMAGE_SIZE),
        tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ]
)


# paths_to_audio = glob.glob('audio/*.wav')
paths_to_audio = glob.glob("../examples/ocean-wave-1.wav")
audio = list()
for path_to_audio in paths_to_audio:
    track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)

    # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
    # thus, the actual time-frequency representation will be visualized
    spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    spec = np.ascontiguousarray(spec.cpu().numpy()).view(np.complex64)
    pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

    audio.append((track, pow_spec))


# paths_to_images = glob.glob('images/*.jpg')
# Read video as a sequence of images using torchvision io
video_path = "../examples/beach.mov"
video_reader = io.read_video(video_path, pts_unit="sec")
video_tensor, audio_tensor, video_info = video_reader
images = [Image.fromarray(frame.numpy()) for frame in video_tensor[-1:]]


def extract_patches_rect(image, patch_size, patches_per_row, patches_per_column):
    patches = []
    width, height = image.size
    stride_x = (width - patch_size) // (patches_per_row - 1)
    stride_y = (height - patch_size) // (patches_per_column - 1)

    for y in range(0, height - patch_size + 1, stride_y):
        for x in range(0, width - patch_size + 1, stride_x):
            patch = ToTensor()((image.crop((x, y, x + patch_size, y + patch_size))))
            patches.append(patch)
    patches = torch.stack(patches, dim=0)
    return patches, stride_x, stride_y


w, h = images[0].size
patch_size = 128
downscale = 32

patches, stride_x, stride_y = extract_patches_rect(
    images[0], patch_size, w // downscale, h // downscale
)
images = patches


# AudioCLIP handles raw audio on input, so the input shape is [batch x channels x duration]
audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio[0:1]])
# standard channel-first shape [batch x channels x height x width]
images = torch.stack([image_transforms(image) for image in images])
# textual input is processed internally, so no need to transform it beforehand
text = [[label] for label in LABELS]


((audio_features, _, _), _), _ = aclp(audio=audio)
# ((_, image_features, _), _), _ = aclp(image=images.to('cuda'))
# Batch up the images to reduce the memory load
aclp = aclp.to("cuda")
image_features = []
for i in tqdm(range(0, images.shape[0], 8), desc="Extracting image features"):
    image_features.append(aclp(image=images[i : i + 8].to("cuda")))
    # move back to CPU to reduce GPU memory usage
    image_features[-1] = image_features[-1][0][0][1].to("cpu")
image_features = torch.cat(image_features, dim=0)

aclp = aclp.to("cpu")
((_, _, text_features), _), _ = aclp(text=text)

audio_features = audio_features / torch.linalg.norm(
    audio_features, dim=-1, keepdim=True
)
image_features = image_features / torch.linalg.norm(
    image_features, dim=-1, keepdim=True
)
text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

new_w = (w - patch_size) // stride_x + 1
new_h = (h - patch_size) // stride_y + 1

scale_audio_image = torch.clamp(aclp.logit_scale_ai.exp(), min=1.0, max=100.0)
scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
scale_image_text = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)

logits_audio_image = scale_audio_image * audio_features @ image_features.T
logits_audio_text = scale_audio_text * audio_features @ text_features.T
logits_image_text = scale_image_text * image_features @ text_features.T

# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(logits_audio_image.cpu().numpy().reshape(new_h, new_w), cmap='jet')
# axs[1].imshow(logits_image_text.cpu().numpy().reshape(new_h, new_w), cmap='jet')
# plt.show()
print(text)
# plt.imshow(logits_audio_text.cpu().numpy().reshape(new_h, new_w), cmap='jet')
# plt.imshow(logits_image_text[:,0].cpu().numpy().reshape(new_h, new_w), cmap='jet')
# plt.show()

plt.imshow(logits_audio_image.cpu().numpy().reshape(new_h, new_w), cmap="jet")
plt.show()
