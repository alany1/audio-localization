from params_proto import Proto, ParamsProto
import wav2clip

# import cv2
from PIL import Image
from features import clip
import torch
import torchvision.io as io
from torchvision.transforms import CenterCrop, Compose


patch_size = {
    "RN50": 32,
    "RN101": 32,
    "RN50x4": None,
    "RN50x16": 8,
    "ViT-B/32": 32,
    "ViT-B/16": 16,
    "ViT-B/8": 8,
    "ViT-L/32": 32,
    "ViT-L/16": 16,
    "ViT-L/8": 8,
    "ViT-H/14": 14,
    "ViT-H/7": 7,
    "ViT-L/14@336px": 14,
}


class VideoArgs(ParamsProto):
    start_frame: int = Proto(help="Start frame of the video.")
    end_frame: int = Proto(help="End frame of the video.")
    source_path: str = Proto(help="Path to the video file.")
    batch_size: int = Proto(help="Batch size for the video.")
    num_frames: int = Proto(
        default=1, help="Number of frames to extract from the video."
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = "ViT-B/32"
    # model = "RN50x64"
    patch_size = patch_size[model]
    high_res = True


def get_frame_embeddings():
    import torch

    # Load the CLIP model
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading CLIP model on device: {}".format(VideoArgs.device))
    model, preprocess = clip.load(VideoArgs.model, device=VideoArgs.device)
    print("Loaded CLIP model on device: {}".format(VideoArgs.device))
    video_tensor, audio, info = io.read_video(VideoArgs.source_path, pts_unit="sec")
    print("Frames count = {}".format(video_tensor.shape[0]))

    # TODO: support arbitrary user specified image sizes? Rather than the resize in the preprocess
    if VideoArgs.high_res:
        # Check there is exactly one center crop transform
        is_center_crop = [isinstance(t, CenterCrop) for t in preprocess.transforms]
        assert (
            sum(is_center_crop) == 1
        ), "There should be exactly one CenterCrop transform"
        # Create new preprocess without center crop
        preprocess = Compose(
            [t for t in preprocess.transforms if not isinstance(t, CenterCrop)]
        )

    images = [Image.fromarray(frame.numpy(), mode="RGB") for frame in video_tensor]
    # Preprocess each image
    preprocessed_images = torch.stack([preprocess(image) for image in images])
    preprocessed_images = preprocessed_images.to(
        VideoArgs.device
    )  # (b, 3, 336, 336) depending on the model preprocessing

    patch_embeddings = []
    for i in range(0, VideoArgs.num_frames, VideoArgs.batch_size):
        batch = preprocessed_images[i : i + VideoArgs.batch_size]
        # Change to half
        batch = batch.half()
        # patch_embeddings.append(model.visual(batch, patch_output=True))
        patch_embeddings.append(model.visual(batch, patch_output=True))
    patch_embeddings = torch.cat(patch_embeddings, dim=0)

    # Reshape patch_embeddings into (N, 24, 24, embedding_dim)
    # patch_size = model.visual.conv1[0].kernel_size[0]
    output_h = preprocessed_images.shape[-2] // VideoArgs.patch_size
    output_w = preprocessed_images.shape[-1] // VideoArgs.patch_size
    patch_embeddings = patch_embeddings.reshape(
        -1, output_h, output_w, patch_embeddings.shape[-1]
    )

    return patch_embeddings, preprocess


def visualize_embeddings(embedding):
    """
    Visualize the first three principal components of embedding to correspond to RGB.
    """
    import matplotlib.pyplot as plt
    import torch
    from sklearn.decomposition import PCA

    # Reshape the embedding into (N, embedding_dim)
    h, w = embedding.shape[1:3]
    embedding = embedding.reshape(-1, embedding.shape[-1])

    # Apply PCA to reduce the dimensionality of the embedding
    pca = PCA(n_components=3)
    pca.fit(embedding.detach().cpu().numpy())
    embedding = torch.from_numpy(pca.transform(embedding.detach().cpu().numpy()))

    # Reshape the embedding back into (N, 24, 24, 3)
    embedding = embedding.reshape(-1, h, w, 3)
    return embedding


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # VideoArgs.source_path = 'examples/ocean-wave-1.wav'
    # print(get_frame_embeddings(VideoArgs.source_path).shape)
    VideoArgs.source_path = "../examples/beach.mov"
    VideoArgs.batch_size = 1
    # Convert video to frames
    # video, audio, info = io.read_video(VideoArgs.source_path, pts_unit='sec')
    # # video_tensor = video.permute(0, 3, 1, 2)
    # video_tensor = video
    # print("Frames count = {}".format(video_tensor.shape[0]))
    # Get embeddings for each frame

    # model = wav2clip.get_model(frame_length=16000, hop_length=16000)

    # frame_pil = Image.fromarray(video_tensor[0].numpy(), mode='RGB')
    embeddings, _ = get_frame_embeddings()

    viz = visualize_embeddings(embeddings).detach().cpu().numpy()
    plt.imshow(viz[0])
    plt.show()
