import cv2
import clip
from PIL import Image
import time
import torch
from utils import *

class VideoFrames():
    """
    Generator class that gets the next frame in specified video
    """
    def __init__(self, source: str):
        self.source = source   # file must be .mp4
        self.success = 1
        self.count = 0
        self.video = self.load_video()

    def load_video(self):
        """
        Loads in the video, returns Video object
        """
        return cv2.VideoCapture(self.source)
    
    def __next__(self):
        """
        Gets next frame in video object
        Returns: Image object of frame if exists, None if final frame reached
        """
        if self.success:
            self.success, frame = self.video.read()
            self.count += 1
            return Image.fromarray(frame)
        else:
            return None

    def __iter__(self):
        return self


def get_patch_embeddings(patch: Image):
    """
    Returns the embeddings of the given patch frame as 1D tensor
    """
    image = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return torch.flatten(image_features)


def get_patch(img_frame: Image, patch_size, x, y) -> Image:
    """
    Gets the square-sized patch of the image centered at
    specified top-left corner
    Returns: Image object of cropped patch
    """
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
    """
    Gets all patch embeddings for the specified image, subsequent
    patches are centered at pixels from the image separated by stride-number
    of pixels. 
    Returns: (N,M,512) sized tensor, dim0=img.height, dim1=img.width, dim2=embedding.size,
            for pixel indices that did not generate a patch to embed, they are filled with
            the embeddings of the patch closest to the top-left
    """
    img_width, img_height = img_frame.size
    img_embeddings = torch.zeros((img_height, img_width, 512))
    for j in range(0, img_height, stride):
        print(j)
        for i in range(0, img_width, stride):
            img_embeddings[j:j+stride, i:i+stride] = get_patch_embeddings(get_patch(img_frame, patch_size, i, j))
    return img_embeddings


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    video_file = FileArgs()
    video_file.source_path = 'scripts/examples/beach.mp4'

    # only testing with first frame of video
    frames = VideoFrames(video_file.source_path)
    first_frame = next(frames)

    # test computing embeddings

    # test using patches
    patch_size = 32
    t_start = time.time()
    frame_embeds = get_image_embeddings(first_frame, 32, 16)
    # save embedding to file
    torch.save(frame_embeds, 'scripts/inputs.t')
    # about 6 min for the single image
    print("total embedding time: ", time.time() - t_start)
    print(frame_embeds.shape)

    # OR test using full image
    # t0 = time.time()
    # frame_clip = get_patch_embeddings(first_frame)
    # t1 = time.time()
    # print(t1-t0)