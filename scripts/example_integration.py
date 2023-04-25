from utils import *
from create_heatmap import *
from get_audio_embeddings import *
from get_frame_embeddings import *


if __name__ == '__main__':    
    audio_file = FileArgs()
    audio_file.source_path = 'scripts/examples/car_ignition.wav'
    audio_clip = get_audio_embeddings(audio_file.source_path)

    video_file = FileArgs()
    video_file.source_path = 'scripts/examples/driving-2.mp4'
    frames = VideoFrames(video_file.source_path)
    first_frame = next(frames)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # test using patches
    patch_size = 64
    # t_start = time.time()
    frame_embeds = get_image_embeddings(first_frame, patch_size, 16, model, preprocess, device)
    torch.save(frame_embeds, 'scripts/driving_embeddings.t')
    print("total embedding time: ", time.time() - t_start)
    print(frame_embeds.shape)

    # frame_embeds = torch.load('scripts/inputs.t')
    beach_heatmap = get_heatmap_probabilities(audio_clip, frame_embeds)
    beach_img = get_heatmap_img(beach_heatmap)
    cv2.imwrite("scripts/driving_heatmap.jpg", beach_img)

    # test using full image
    # t0 = time.time()
    # frame_clip = get_patch_embeddings(first_frame)
    # t1 = time.time()
    # print(t1-t0)
    # similarity = 100.0 * frame_clip @ audio_clip
    # print(frame_clip.shape)
    # print(audio_clip.shape)
    # print(similarity)

    # rocks_clip = get_patch_embeddings(Image.open("scripts/examples/rocks.jpg"))    
    # similarity = 100.0 * rocks_clip @ audio_clip
    # print(similarity)

    # cos = CosineSimilarity(dim=1, eps=1e-6)
    # r_similarity = cos(rocks_clip, torch.tensor(audio_clip))
    # print(r_similarity)

    # c_similarity = cos(frame_clip, torch.tensor(audio_clip))
    # print(c_similarity)