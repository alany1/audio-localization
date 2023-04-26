import torch
from matplotlib import pyplot as plt
from params_proto import Proto, ParamsProto, PrefixProto
from frames import FrameArgs, get_frame_embeddings
from audio import AudioArgs, get_audio_embeddings
from scripts.AudioCLIP.model import AudioCLIP

class Args(PrefixProto):
    path = 'AudioCLIP-Full-Training.pt'
    model = AudioCLIP(pretrained=f'assets/{path}')
    model.eval()

def show_heatmap(index, logits_audio_image, images, new_w, new_h, cmap='jet'):
    logits = logits_audio_image[index].reshape(new_w, new_h).detach().cpu().numpy()
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(images[index])
    axs[1].imshow(logits, cmap=cmap)
    plt.show()

def main(text_features=None):
    print("Getting image embeddings...")
    image_features, new_w, new_h, images = get_frame_embeddings(Args.model)

    if text_features is None:
        print("Getting audio embeddings...")
        audio_features = get_audio_embeddings(Args.model).to('cpu')
    else:
        print("Using provided text features...")
        audio_features = text_features.to('cpu')

    # get the heatmap now
    scale_audio_image = torch.clamp(Args.model.logit_scale_ai.exp(), min=1.0, max=100.0).to('cpu')
    scale_image_text = torch.clamp(Args.model.logit_scale.exp(), min=1.0, max=100.0).to('cpu')

    if text_features is None:
        logits_audio_image = scale_audio_image * audio_features @ image_features.T
    else:
        logits_audio_image = scale_image_text * audio_features @ image_features.T

    return logits_audio_image, new_w, new_h, images

if __name__ == "__main__":

    FrameArgs.video_path = "../examples/driving-2.mp4"
    FrameArgs.patch_size = 128
    FrameArgs.downscale = 32

    AudioArgs.path = "../examples/car-ignition.wav"

    # Text
    text = ['ocean']
    text = [[label] for label in text]
    ((_, _, text_features), _), _ = Args.model(text=text)

    logits_audio_image, new_w, new_h, images = main(text_features)
    show_heatmap(0, logits_audio_image, images, new_h, new_w)


