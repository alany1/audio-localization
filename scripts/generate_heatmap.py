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
    target_feat = torch.tensor(
        get_audio_embeddings(AudioArgs.audio_path), device=VideoArgs.device
    )
    embedding, preprocess = get_frame_embeddings()

    # Compute cosine similarity between each pixel of embedding and target_feat
    heatmap = F.cosine_similarity(embedding, target_feat, dim=-1)

    # Normalize heatmap
    heatmap = heatmap / torch.max(heatmap)

    return heatmap, preprocess

    # heatmap = torch.einsum("ijk,kl->ijl", embedding, target_feat)
    # heatmap = F.normalize(heatmap, dim=2)
    # heatmap = torch.einsum("ijk,ijk->ij", heatmap, heatmap)
    # heatmap = heatmap / torch.max(heatmap)
    # return heatmap


if __name__ == "__main__":
    from PIL import Image
    from get_audio_embeddings import AudioArgs, get_audio_embeddings
    from get_frame_embeddings import VideoArgs, get_frame_embeddings
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import torchvision.io as io
    import torch
    import torchvision

    AudioArgs.audio_path = "examples/car-ignition.wav"
    VideoArgs.source_path = "examples/driving-2.mp4"

    # AudioArgs.audio_path = 'examples/ocean-wave-1.wav'
    # VideoArgs.source_path = 'examples/beach.mov'

    VideoArgs.num_frames = 145
    VideoArgs.batch_size = 1

    hm, preprocess = makeHeatmap()
    hm = hm.detach().cpu().numpy()
    n_frames = len(hm)
    h, w = hm[0].shape

    # $ Save a video of the frames as well
    video_tensor, audio, info = io.read_video(VideoArgs.source_path, pts_unit="sec")
    # Resize and center crop video_tensor to 224x224
    # video_tensor = torch.nn.functional.interpolate(video_tensor.permute(0, 3, 1, 2), size=(h, w), mode='bilinear').permute(0, 2, 3, 1)

    io.write_video(
        "results/output_frames.mp4", video_tensor, fps=info["video_fps"]
    )  # , video_codec='libx264', options={'crf': 18} , audio_array=None, audio_fps=44100)

    # Convert to RGB using Jet colormap
    color_images = []
    for img in hm:
        img_norm = cv2.normalize(
            img, None, 0, 255, cv2.NORM_MINMAX
        )  # Normalize pixel values to 0-255
        color_img = cv2.applyColorMap(img_norm.astype(np.uint8), cv2.COLORMAP_JET)
        color_images.append(color_img)

    # Define the video writer
    width, height = w, h
    fps = info["video_fps"]
    output_file = "results/output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write the color images to the video file
    for img in color_images:
        video_writer.write(img)

    # Release the video writer and close the file
    video_writer.release()

    # Overlay videos on top of each other with transparency for decreasing heatmap
