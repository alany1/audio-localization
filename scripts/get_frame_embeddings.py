from params_proto import Proto, ParamsProto
import wav2clip
# import cv2
from PIL import Image
from features import clip

import torchvision.io as io
class VideoArgs(ParamsProto):
    start_frame: int = Proto(help='Start frame of the video.')
    end_frame: int = Proto(help='End frame of the video.')
    source_path: str = Proto(help='Path to the video file.')
    batch_size: int = Proto(help='Batch size for the video.')
def load(model_str, device, root):
    """
    Loads the model from the given path.
    """
    assert model_str in ["ViT-B/32"], "Model not supported."
    print("Loading model from path: {}".format(model_str))

    model = torch.hub.load(root, model_str, pretrained=True)
    model.to(device)
    model.eval()

    return model
def get_frame_embedding(images):
    import torch
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading CLIP model on device: {}".format(device))
    model, preprocess = clip.load('ViT-L/14@336px', device=device)
    print("Loaded CLIP model on device: {}".format(device))

    # Preprocess each image
    # start_time = time.perf_counter()
    # with logger.time("clip_preprocess_images"):
    preprocessed_images = torch.stack([preprocess(image) for image in images])
    preprocessed_images = preprocessed_images.to(device)  # (b, 3, 336, 336) depending on the model preprocessing

    patch_embeddings = []
    for i in range(0, len(images), VideoArgs.batch_size):
        batch = preprocessed_images[i: i + VideoArgs.batch_size]
        # Change to half
        batch = batch.half()
        # patch_embeddings.append(model.visual(batch, patch_output=True))
        patch_embeddings.append(model.visual(batch, patch_output = True))
    patch_embeddings = torch.cat(patch_embeddings, dim=0)

    # Reshape patch_embeddings into (N, 24, 24, embedding_dim)
    patch_embeddings = patch_embeddings.reshape(-1, 24, 24, patch_embeddings.shape[-1])

    return patch_embeddings

def visualize_embeddings(embedding):
    """
    Visualize the first three principal components of embedding to correspond to RGB.
    """
    import matplotlib.pyplot as plt
    import torch
    from sklearn.decomposition import PCA

    # Reshape the embedding into (N, embedding_dim)
    embedding = embedding.reshape(-1, embedding.shape[-1])

    # Apply PCA to reduce the dimensionality of the embedding
    pca = PCA(n_components=3)
    pca.fit(embedding.detach().cpu().numpy())
    embedding = torch.from_numpy(pca.transform(embedding.detach().cpu().numpy()))

    # Reshape the embedding back into (N, 24, 24, 3)
    embedding = embedding.reshape(-1, 24, 24, 3)
    return embedding



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # VideoArgs.source_path = 'examples/ocean-wave-1.wav'
    # print(get_frame_embeddings(VideoArgs.source_path).shape)
    VideoArgs.source_path = 'examples/beach.mov'
    VideoArgs.batch_size = 1
    # Convert video to frames
    video, audio, info = io.read_video(VideoArgs.source_path, pts_unit='sec')
    # video_tensor = video.permute(0, 3, 1, 2)
    video_tensor = video
    print("Frames count = {}".format(video_tensor.shape[0]))
    # Get embeddings for each frame

    model = wav2clip.get_model(frame_length=16000, hop_length=16000)

    frame_pil = Image.fromarray(video_tensor[0].numpy(), mode='RGB')
    embeddings = get_frame_embedding([frame_pil])

    viz = visualize_embeddings(embeddings).detach().cpu().numpy()

