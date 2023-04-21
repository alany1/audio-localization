# from get_audio_embeddings import AudioArgs, get_audio_embeddings
# from get_frame_embeddings import VideoArgs, get_frame_embeddings
def makeHeatmap():
    # Compute cosine similarity between target_feat and embedding for each point to generate heatmap
    # embedding: (N, 24, 24, 512)
    # target_feat: (512)
    # heatmap: (N, 24, 24)
    import torch
    import torch.nn.functional as F
    # Compute cosine similarity between target_feat and embedding for each point to generate heatmap
    target_feat = torch.tensor(get_audio_embeddings(AudioArgs.audio_path), device = VideoArgs.device)
    embedding = get_frame_embeddings()[0]

    # Compute cosine similarity between each pixel of embedding and target_feat
    heatmap = F.cosine_similarity(embedding, target_feat, dim=-1)

    # Normalize heatmap
    heatmap = heatmap / torch.max(heatmap)

    return heatmap


    # heatmap = torch.einsum("ijk,kl->ijl", embedding, target_feat)
    # heatmap = F.normalize(heatmap, dim=2)
    # heatmap = torch.einsum("ijk,ijk->ij", heatmap, heatmap)
    # heatmap = heatmap / torch.max(heatmap)
    # return heatmap

if __name__ == '__main__':
    from PIL import Image
    from get_audio_embeddings import AudioArgs, get_audio_embeddings
    from get_frame_embeddings import VideoArgs, get_frame_embeddings
    import matplotlib.pyplot as plt

    AudioArgs.audio_path = 'examples/ocean-wave-1.wav'
    VideoArgs.source_path = 'examples/beach.mov'
    VideoArgs.batch_size = 1

    makeHeatmap()
