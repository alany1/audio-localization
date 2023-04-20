from params_proto import Proto, ParamsProto
import wav2clip
import cv2
import torchvision.io as io
class VideoArgs(ParamsProto):
    start_frame: int = Proto(help='Start frame of the video.')
    end_frame: int = Proto(help='End frame of the video.')
    source_path: str = Proto(help='Path to the video file.')

if __name__ == '__main__':
    # VideoArgs.source_path = 'examples/ocean-wave-1.wav'
    # print(get_frame_embeddings(VideoArgs.source_path).shape)
    VideoArgs.source_path = 'examples/beach.mov'
    # Convert video to frames
    video, audio, info = io.read_video(VideoArgs.source_path, pts_unit='sec')
    video_tensor = video.permute(3, 0, 1, 2)
    print(video_tensor.shape)

    # Get embeddings for each frame

    model = wav2clip.get_model(frame_length=16000, hop_length=16000)
    embeddings = wav2clip.embed_audio(audio, model)