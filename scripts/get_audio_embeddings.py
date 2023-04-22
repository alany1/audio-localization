import wav2clip
from params_proto import Proto, ParamsProto
import torch, torchaudio
import numpy as np
import cv2
import clip
from PIL import Image
from torch.nn import CosineSimilarity
import time
import matplotlib
import matplotlib.pyplot as plt

class FileArgs(ParamsProto):
    source_path: str = Proto(help='Path to the audio file.')


class VideoFrames():
    def __init__(self, source):
        self.source = source
        self.success = 1
        self.count = 0
        self.video = self.load_video()

    def load_video(self):
        return cv2.VideoCapture(self.source)
    
    def __next__(self):
        if self.success:
            self.success, frame = self.video.read()
            self.count += 1
            return Image.fromarray(frame)
        else:
            return None

    def __iter__(self):
        return self


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


def get_patch_embeddings(patch: Image):
    """
    Returns the embeddings of the given patch frame
    """
    image = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return torch.flatten(image_features)


def get_patch(img_frame: Image, patch_size, x, y):
    img_width, img_height = img_frame.size
    half_dim = patch_size // 2
    left = 0 if x < 0 else x - half_dim
    top = 0 if y < 0 else y - half_dim
    right = x + half_dim
    right = img_width if right >= img_width else right
    bottom = y + half_dim
    bottom = img_height if bottom >= img_height else bottom
    return img_frame.crop((left, top, right, bottom))


def get_image_embeddings(img_frame: Image, patch_size, stride):
    img_width, img_height = img_frame.size
    img_embeddings = torch.zeros((img_height, img_width, 512))
    for j in range(0, img_height, stride):
        print(j)
        for i in range(0, img_width, stride):
            img_embeddings[j:j+stride, i:i+stride] = get_patch_embeddings(get_patch(img_frame, patch_size, i, j))
    return img_embeddings


def get_similarity(audio, img):
    return img @ audio


def get_image_heatmap(audio_embedding, img_embeddings):
    heatmap = get_similarity(audio_embedding, img_embeddings)
    heatmap = heatmap / torch.max(heatmap)
    return heatmap


def plot_heatmap(heatmap, image: Image):
    a_component = heatmap[:,:,1]
    th = cv2.threshold(a_component,140,255,cv2.THRESH_BINARY)[1]
    blur = cv2.GaussianBlur(th,(13,13), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)
    super_imposed_img.show()



if __name__ == '__main__':
    # brain dump, will split into files and all later
    audio_file = FileArgs()
    audio_file.source_path = 'scripts/examples/ocean-wave-1.wav'

    video_file = FileArgs()
    video_file.source_path = 'scripts/examples/beach.mp4'

    audio_clip = get_audio_embeddings(audio_file.source_path)
    frames = VideoFrames(video_file.source_path)
    first_frame = next(frames)
    print(first_frame.size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # test using patches
    patch_size = 32
    # t_start = time.time()
    # frame_embeds = get_image_embeddings(first_frame, 32, 16)
    # torch.save(frame_embeds, 'scripts/inputs.t')
    # print("total embedding time: ", time.time() - t_start)
    # print(frame_embeds.shape)

    frame_embeds = torch.load('scripts/inputs.t')
    frame_heatmap = get_image_heatmap(audio_clip, frame_embeds)
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(frame_heatmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    # plot_heatmap(frame_heatmap, first_frame)

    # test using full image
    # t0 = time.time()
    # frame_clip = get_patch_embeddings(first_frame)
    # t1 = time.time()
    # print(t1-t0)
    # similarity = 100.0 * frame_clip @ audio_clip
    # print(frame_clip.shape)
    # print(audio_clip.shape)
    # print(similarity)

    # rocks_clip = get_patch_embeddings(Image.open("scripts/examples/rocks.jpg"))    
    # similarity = 100.0 * rocks_clip @ audio_clip
    # print(similarity)

    # cos = CosineSimilarity(dim=1, eps=1e-6)
    # r_similarity = cos(rocks_clip, torch.tensor(audio_clip))
    # print(r_similarity)

    # c_similarity = cos(frame_clip, torch.tensor(audio_clip))
    # print(c_similarity)

    
