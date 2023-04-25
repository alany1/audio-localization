import torch
import torchaudio
import torchvision.transforms as transforms
import torchvision.transforms.functional
from PIL.Image import Image
from torchvision import io
from tqdm import tqdm

from model.audioclip import AudioCLIP

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from ml_logger import logger

class FrameArgs:
    video = "../examples/beach.mov"
    output = "examples/beach.pt"
    audio = None


def load_audio_clip_model():
    model_path = "assets/AudioCLIP-Full-Training.pt"
    model = AudioCLIP(pretrained=model_path)
    model.eval()
    model.to("cuda")
    return model

def extract_patches(image, patch_size, stride):
    patches = []
    width, height = image.size

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches

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


def get_embeddings(model, patch_tensors):
    # transform = Compose([
    #     Resize((224, 224)),
    #     ToTensor(),
    #     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    #
    # print("Getting embeddings for", len(patches), "patches on device", model.device)
    #
    # patch_tensors = [transform(patch) for patch in patches]
    # patch_tensors = torch.stack(patch_tensors).to(model.device)
    #
    with torch.no_grad():
        # batch them up to reduce memory usage
        batch_size = 1024
        embeddings = []
        for i in tqdm(range(0, len(patch_tensors), batch_size), desc='Batch number'):
            batch = patch_tensors[i:i+batch_size]
            embeddings.append(model.encode_image(batch))
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    return torch.tensor(embeddings)

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


def main():
    from matplotlib import pyplot as plt
    model = load_audio_clip_model()

    video_path = FrameArgs.video
    video_reader = io.read_video(video_path, pts_unit='sec')
    video_tensor, audio_tensor, video_info = video_reader
    pil_frames = [Image.fromarray(frame.numpy()) for frame in video_tensor[:1]]

    # image = pil_frames[100]
    # image = Image.open(image_path).convert("RGB")
    w,h = pil_frames[0].size
    patch_size = 128
    downscale = 32

    # patches = extract_patches(image, patch_size, stride)
    all_embeddings = []
    all_images = []
    transform = Compose([
        Resize((224, 224)),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # frames = torch.stack([torch.nn.functional.to_tensor(im) for im in pil_frames])
    all_patches = []
    for i in tqdm(range(len(pil_frames)), desc="Extracting patches"):
        patches, stride_x, stride_y = extract_patches_rect(pil_frames[i], patch_size, w//downscale, h//downscale)
        # all_patches.append([transform(x) for x in patches])
        all_patches.append(transform(patches))

    print('Finished extracting patches', len(all_patches))
    all_patches = torch.cat(all_patches, dim=0).to(model.device)
    all_embeddings = get_embeddings(model, all_patches)
    print('Finished extracting embeddings', all_embeddings.shape)


    # for i, image in enumerate(pil_frames):
    #     patches, stride_x, stride_y = extract_patches_rect(image, patch_size, w//downscale, h//downscale)
    #     embeddings = get_embeddings(model, patches)
    #     all_embeddings.append(embeddings)
    #     print("Finished extracting embeddings for image ", i)
    # all_embeddings = torch.cat(all_embeddings, dim=0)
    # Visualize the embeddings
    # For this image, reshape back into the original resolution proportions
    # embeddings = embeddings.reshape((w//stride-1, h//stride-1, 1024))
    new_w = (w-patch_size)//stride_x + 1
    new_h = (h-patch_size)//stride_y + 1
    for image in pil_frames:
        all_images.append(image.resize((new_w, new_h)))
    # embeddings = embeddings.reshape((-1, new_h, new_w, 1024))
    all_embeddings = all_embeddings.reshape((-1, new_h, new_w, 1024))
    return all_embeddings, all_images
    # Then visualize the first three principal components
    embeddings = visualize_embeddings(all_embeddings[0])
    # Convert to PIL
    embeddings = transforms.ToPILImage()(embeddings[0].permute(2,0,1))

    fig, axs = plt.subplots(1,2)

    axs[0].imshow(image.resize((new_w, new_h)))
    axs[1].imshow(embeddings)

    plt.show()

if __name__ == "__main__":
    logger.configure(prefix="/alanyu/audio-localization/test")


    FrameArgs.output = "backup.pt"
    # FrameArgs.video = "../examples/beach.mov"
    FrameArgs.video = "../examples/driving-2.mp4"
    frame,image = main()

    # from get_audio_embeddings import Result
    if not FrameArgs.audio:
        audio1 = torch.load("../examples/ignition.pt")
        # audio2 = torch.load("../examples/dirt.pt")
        # get l2 norm difference between the two audios
        # diff = np.linalg.norm(audio1-audio2)
        # print("L2 norm difference between the two audios: ", diff)
        audio = audio1
    else:
        raise NotImplementedError("Need to implement audio loading")

    # Plot torch cosine similarity between frame and audio
    from torch.nn.functional import cosine_similarity
    from matplotlib import pyplot as plt

    # Plot cosine similarity between frame and audio along the 1024 dimension

    fig, axs = plt.subplots(1,2)

    axs[0].imshow(cosine_similarity(frame[0], audio, dim=-1).numpy(), cmap='jet')
    axs[1].imshow(image[0])
    # plt.imshow(cosine_similarity(frame, audio, axis=-1).numpy())
    # Use the Jet heatmap
    # plt.set_cmap('jet')
    # plt.colorbar()
    plt.show()

