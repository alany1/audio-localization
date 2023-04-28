import os

import torch
import torchvision
from PIL import Image
from params_proto import Proto, ParamsProto, PrefixProto
from torchvision import io
from torchvision.transforms import ToTensor
from tqdm import tqdm

from scripts.AudioCLIP.model import AudioCLIP
import time
import shutil

class FrameArgs(PrefixProto):
    IMAGE_SIZE = 224 # derived from CLIP, how to upscale the patches basically
    IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
    IMAGE_STD = 0.26862954, 0.26130258, 0.27577711
    # IMAGE_MEAN = .5,.5,.5
    # IMAGE_STD = .5,.5,.5

    device = "cuda"
    video_path = "../examples/beach.mov"
    patch_size = 128
    downscale = 32


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

def visualize_embeddings(embedding):
    """
    Visualize the first three principal components of embedding to correspond to RGB.
    """
    import matplotlib.pyplot as plt
    import torch
    from sklearn.decomposition import PCA

    # Reshape the embedding into (N, embedding_dim)
    h,w = embedding.shape[:-1]
    embedding = embedding.reshape(-1, embedding.shape[-1])

    # Apply PCA to reduce the dimensionality of the embedding
    pca = PCA(n_components=3)
    pca.fit(embedding.detach().cpu().numpy())
    embedding = torch.from_numpy(pca.transform(embedding.detach().cpu().numpy()))

    # Reshape the embedding back into (N, 24, 24, 3)
    embedding = embedding.reshape(-1, w, h, 3)
    return embedding


def get_frame_embeddings(model):
    """
    Currently only supports a single frame
    TODO: play with batch size.
    """
    image_transforms = torchvision.transforms.Compose([
        # tv.transforms.ToTensor(),
        torchvision.transforms.Resize(FrameArgs.IMAGE_SIZE, interpolation=Image.BICUBIC, antialias=True),
        torchvision.transforms.CenterCrop(FrameArgs.IMAGE_SIZE),
        torchvision.transforms.Normalize(FrameArgs.IMAGE_MEAN, FrameArgs.IMAGE_STD)
    ])
    model.to(FrameArgs.device)
    if FrameArgs.video_path.split(".")[-1] == "jpeg":
        print("Processing the image", FrameArgs.video_path)
        image = Image.open(FrameArgs.video_path)
        images = [image]
    else:
        print("Processing the video", FrameArgs.video_path)
        video_reader = io.read_video(FrameArgs.video_path, pts_unit='sec')
        video_tensor, audio_tensor, video_info = video_reader
        images = [Image.fromarray(frame.numpy()) for frame in video_tensor[-1:]]

    w, h = images[0].size
    # time this line
    t0 = time.time()
    patches, stride_x, stride_y = extract_patches_rect(images[0], FrameArgs.patch_size, w // FrameArgs.downscale, h // FrameArgs.downscale)
    print(f"Extracting patches took {time.time() - t0} seconds")

    all_patches = torch.stack([image_transforms(patch) for patch in patches])

    image_features = []
    for i in tqdm(range(0, all_patches.shape[0], 8), desc="Extracting image features"):
        image_features.append(model(image=all_patches[i:i + 8].to(FrameArgs.device)))
        # move back to CPU to reduce GPU memory usage
        image_features[-1] = image_features[-1][0][0][1].detach().cpu()
    image_features = torch.cat(image_features, dim=0)

    image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)

    new_w = (w - FrameArgs.patch_size) // stride_x + 1
    new_h = (h - FrameArgs.patch_size) // stride_y + 1

    return image_features, new_w, new_h, images


def save_frame_embeddings(model, num_frames=1, tmp_dir="/tmp/frames"):
    """
    To circumvent memory issues, we write the features to disk in a temporary directory for later use.

    We return the path to the temporary directory along with (new_h, new_w, images).
    """
    model.to(FrameArgs.device)
    image_transforms = torchvision.transforms.Compose([
        # tv.transforms.ToTensor(),
        torchvision.transforms.Resize(FrameArgs.IMAGE_SIZE, interpolation=Image.BICUBIC, antialias=True),
        torchvision.transforms.CenterCrop(FrameArgs.IMAGE_SIZE),
        torchvision.transforms.Normalize(FrameArgs.IMAGE_MEAN, FrameArgs.IMAGE_STD)
    ])

    video_reader = io.read_video(FrameArgs.video_path, pts_unit='sec')
    video_tensor, audio_tensor, video_info = video_reader
    images = [Image.fromarray(frame.numpy()) for frame in video_tensor]

    w, h = images[0].size

    print("Removing old temporary directory")
    shutil.rmtree(tmp_dir)
    print("Creating new temporary directory")
    os.mkdir(tmp_dir)

    # time this line
    for x in tqdm(range(num_frames), desc="Extracting image features for each frame 🎥📸", colour="green"):
        t0 = time.time()
        patches, stride_x, stride_y = extract_patches_rect(images[x], FrameArgs.patch_size, w // FrameArgs.downscale,
                                                           h // FrameArgs.downscale)
        if x%10==0:
            print(f"Extracting patches for frame {x+1} took {time.time() - t0} seconds")

        all_patches = torch.stack([image_transforms(patch) for patch in patches])

        image_features = []
        for i in range(0, all_patches.shape[0], 8):
            image_features.append(model(image=all_patches[i:i + 8].to(FrameArgs.device)))
            # move back to CPU to reduce GPU memory usage
            image_features[-1] = image_features[-1][0][0][1].detach().cpu()
        image_features = torch.cat(image_features, dim=0)

        image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)

        # save the features into the temporary directory
        torch.save(image_features, os.path.join(tmp_dir, f"frame_{x}.pt"))

    new_w = (w - FrameArgs.patch_size) // stride_x + 1
    new_h = (h - FrameArgs.patch_size) // stride_y + 1

    return tmp_dir, new_w, new_h, images



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    FrameArgs.video_path = "../examples/beach.mov"
    aclp = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}')
    aclp.eval()

    embeddings, new_w, new_h, images = get_frame_embeddings(aclp)
    # tmp_dir, new_w, new_h, images = save_frame_embeddings(aclp, num_frames=100)

    pre_viz = embeddings.reshape(new_w, new_h, -1)
    viz = visualize_embeddings(pre_viz)
    #
    # # normalize
    viz = (viz - viz.min()) / (viz.max() - viz.min())

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(viz[0])
    axs[0].set_title("PCA Visualization")
    axs[1].imshow(images[0])
    axs[1].set_title("Original Frame")
    plt.show()