import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from params_proto import Proto, ParamsProto, PrefixProto
from frames import FrameArgs, get_frame_embeddings, save_frame_embeddings, process_frames
from audio import AudioArgs, get_audio_embeddings
from scripts.AudioCLIP.model import AudioCLIP
from tqdm import tqdm

class Args(PrefixProto):
    path = "AudioCLIP-Full-Training.pt"
    model = AudioCLIP(pretrained=f"assets/{path}")
    model.eval()


def show_heatmap(index, logits_audio_image, images, new_w, new_h, cmap="jet"):
    logits = logits_audio_image[index].reshape(new_w, new_h).detach().cpu()
    logits = torch.nn.functional.interpolate(
        logits.unsqueeze(0).unsqueeze(0),
        size=images[index].size[::-1],
        mode="bicubic",
        align_corners=True,
    ).numpy()[0, 0]
    # logits = logits_audio_image[index].reshape(new_w, new_h).detach().cpu().numpy()

    # upsample logits to match image resolution
    # logits = logits.repeat(4, axis=0).repeat(4, axis=1)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(images[index])
    # axs[1].imshow(logits, cmap=cmap)

    # Overlay logits with cmap=cmap and the images[index] on axs[2]
    axs[1].imshow(images[index])
    axs[1].imshow(logits, cmap=cmap, alpha=0.3)

    plt.show()


def main(text_features=None):
    print("Getting image embeddings...")
    image_features, new_w, new_h, images = get_frame_embeddings(Args.model)

    print("text_features is", text_features)
    if text_features is None:
        print("Getting audio embeddings...")
        audio_features = get_audio_embeddings(Args.model).to("cpu")
    else:
        print("Using provided text features...")
        audio_features = text_features.to("cpu")

    # get the heatmap now
    scale_audio_image = torch.clamp(
        Args.model.logit_scale_ai.exp(), min=1.0, max=100.0
    ).to("cpu")
    scale_image_text = torch.clamp(Args.model.logit_scale.exp(), min=1.0, max=100.0).to(
        "cpu"
    )

    if text_features is None:
        logits_audio_image = scale_audio_image * audio_features @ image_features.T
    else:
        logits_audio_image = scale_image_text * audio_features @ image_features.T

    return logits_audio_image, new_w, new_h, images

def save_movie_overlay(heatmap, images, num_frames, cmap='jet', sample_factor=5):
    """
    Save movie as heatmap overlayed on top of images.
    """
    from PIL import Image
    # for each image, overlay the heatmap on top of it and use it as a frame in output video
    output_frames = []
    for i in tqdm(range(min(len(images), num_frames))):
        fig, ax = plt.subplots()
        ax.imshow(images[i])
        ax.imshow(heatmap[sample_factor*(i//sample_factor)], cmap=cmap, alpha=0.3)
        canvas = fig.canvas
        canvas.draw()
        output_frames.append(Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb()))

    plt.close()

    return output_frames



if __name__ == "__main__":

    FrameArgs.video_path = "../examples/cars_driving_by_2.mov"
    # FrameArgs.video_path = "../examples/violin-2.jpeg"
    # FrameArgs.video_path = "../examples/beach.mov"
    FrameArgs.patch_size = 128
    FrameArgs.downscale = 32

    # AudioArgs.path = "../examples/violin-sound.wav"
    AudioArgs.path = "../examples/car-ignition.wav"
    # AudioArgs.path = "../examples/ocean-wave-1.wav"

    audio_features = get_audio_embeddings(Args.model)

    # Text
    text = ["driving car"]
    text = [[label] for label in text]
    print("Getting text embeddings...")
    ((_, _, text_features), _), _ = Args.model(text=text)

    # Take the average of audio and text
    # source_features = (audio_features + text_features) / 2
    source_features = audio_features
    num_frames = 455
    tmp_dir, new_w, new_h, images = save_frame_embeddings(Args.model, num_frames=num_frames, tmp_dir='/tmp/car_full_2', skip=True)
    movie = process_frames(tmp_dir, source_features, new_w, new_h, images, num_frames=num_frames) #, supervision_feature=text_features)

    scale_image_text = torch.clamp(Args.model.logit_scale.exp(), min=1.0, max=100.0).to("cpu")
    scale_audio_image = torch.clamp(Args.model.logit_scale_ai.exp(), min=1.0, max=100.0).to("cpu")

    # movie = [(scale_image_text * frame).detach() for frame in movie]
    movie = [(scale_audio_image * scale_image_text * frame).detach() for frame in movie]
    # convert to pil image with cmap jet
    movie = [torch.nn.functional.interpolate(
        frame.unsqueeze(0).unsqueeze(0),
        size=images[0].size[::-1],
        mode="bicubic",
        align_corners=True,
    ).numpy()[0, 0] for frame in movie]

    sample_factor = 2
    output_movie = save_movie_overlay(movie, images, num_frames=num_frames, sample_factor=sample_factor)

    # Write to movie file
    print("Writing movie to movie.mp4")
    with imageio.get_writer('movie.mp4', fps=30) as writer:
        for i, pil_image in enumerate(output_movie):
            # convert the PIL image to a numpy array
            numpy_image = np.array(pil_image)
            # write the numpy array to the movie file
            writer.append_data(numpy_image)

    # logits_audio_image, new_w, new_h, images = main()
    # logits_image_text, new_w, new_h, images = main(text_features)
    #
    # logits = logits_audio_image * logits_image_text
    # # logits = logits_audio_image
    # # logits = logits_image_text
    # # Normalize logits
    # logits = logits / torch.linalg.norm(logits, dim=-1, keepdim=True)
    #
    # show_heatmap(0, logits, images, new_h, new_w)
