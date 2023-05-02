import cv2
import torch
from PIL import Image
import time
import numpy as np
from utils import *
from get_audio_embeddings import *
from get_frame_embeddings import *

def get_heatmap_probabilities(audio_embedding, img_embeddings):
    """
    Returns normalized probability heatmap of each pixel similarity of the image
    to the audio
    """
    heatmap = get_similarity(audio_embedding, img_embeddings)
    print("max similarity: ", torch.max(heatmap))
    heatmap = heatmap / torch.max(heatmap)
    return heatmap


def get_heatmap_img(heatmap):
    """
    Generates the colored image of the heatmap
    """
    img_map = 255 * heatmap
    img_map = img_map.to(torch.uint8)
    img_map = torch.unsqueeze(img_map, axis=2)
    img = cv2.applyColorMap(img_map.numpy(), cv2.COLORMAP_JET)
    return img


def overlay_heatmap(img, heatmap):
    """
    Overlays the heatmap on top of the specified image
    """
    hm_img = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    return Image.blend(img, hm_img, 0.5)

if __name__ == '__main__':
    # test image generation on a random heatmap
    test = torch.tensor(np.random.random((100, 100)))
    test_img = get_heatmap_img(test)
    cv2.imwrite("scripts/test_heatmap.jpg", test_img)

    # test for the beach frame
    loaded_embeddings = torch.load('scripts/inputs.t')
    audio_file = FileArgs()
    audio_file.source_path = 'scripts/examples/ocean-wave-1.wav'
    audio_clip = get_audio_embeddings(audio_file.source_path)

    # makes heatmap
    beach_heatmap = get_heatmap_probabilities(audio_clip, loaded_embeddings)
    beach_img = get_heatmap_img(beach_heatmap)
    beach_img = gaussian_blur(beach_img)
    cv2.imwrite("scripts/blurred_heatmap.jpg", beach_img)

    # overlays heatmap onto original beach frame image
    video_file = FileArgs()
    video_file.source_path = 'scripts/examples/beach.mp4'
    frames = VideoFrames(video_file.source_path)
    first_frame = next(frames)
    blended = overlay_heatmap(first_frame, beach_img)
    blended.save("scripts/overlay_heatmap.jpg")


    