# CLAP: CLIP Localization in Audio-Visual Pairings

This repository contains the code implementation for our project [CLAP: CLIP Localization in Audio-Visual Pairings](https://alany1.github.io/audio/index.html).

<img src="assets/demos.gif" max-width="100%" height="auto" />

## Contents
1. [Setup](#setup)
2. [Extracting Audio, Video, and Text Features](#features)
3. [Localizing the Features](#integrate)
4. [Examples](#examples)

## Setup <a name="setup"></a>
Install dependencies in a new conda environment:
```
conda create -n clap python=3.7
conda activate clap
pip install -r requirements.txt
```

To run the model, you will need to download the state dictionary of CLIP, AudioCLIP, and other related files. You can download the pre-trained model from [here](https://github.com/AndreyGuzhov/AudioCLIP). Place the model in the `scripts/AudioCLIP/assets` folder. 

## Extracting Audio, Video, and Text Features <a name="features"></a>

First, load the AudioCLIP model.
```
from scripts.AudioCLIP.model import AudioCLIP
model = AudioCLIP("path/to/model")
```

Set the path to your audio file in the `AudioArgs` class. Then, make a call to `get_audio_embeddings` to extract the audio features. 
```
from scripts.AudioCLIP.args import AudioArgs

AudioArgs.audio_file = 'path/to/audio/file'
audio_features = get_audio_embeddings(model)
```

(Optional) Next, extract some text features.
```
text = [['text1'], ['text2'], ['text3']]
((_, _, text_features), _), _ = Args.model(text=text)
```

To extract visual features, set your video path in the `FrameArgs` class. We then extract video features and save them to a temporary directory. To accelerate feature extraction, pass in a value for downscale in the scale parameter.
```
from scripts.FrameCLIP.args import FrameArgs

num_frames = ... # number of frames to extract
tmp_dir, new_w, new_h, images = save_frame_embeddings(
        model, num_frames=num_frames, tmp_dir="/tmp/my_movie/", scale=4
    )
```

## Localizing the Features <a name="integrate"></a>

Pass in the directory containing the temporary directory containing the video features.
```
# Compute similarity scores and produce heatmap
movie = process_frames(
        tmp_dir, source_features, new_w, new_h, images, num_frames=num_frames
    )


# Upsample the movie to the original resolution
movie = [
    torch.nn.functional.interpolate(
        frame.unsqueeze(0).unsqueeze(0),
        size=images[0].size[::-1],
        mode="bicubic",
        align_corners=True,
    ).numpy()[0, 0]
    for frame in movie
]

# Save the movie
output_movie = save_movie_overlay(
        movie, images, num_frames=num_frames
    )
    
print("Writing movie to movie.mp4")
with imageio.get_writer("movie.mp4", fps=30) as writer:
    for i, pil_image in enumerate(output_movie):
        # convert the PIL image to a numpy array
        numpy_image = np.array(pil_image)
        # write the numpy array to the movie file
        writer.append_data(numpy_image)
```

## Examples <a name="examples"></a>
For examples of results, please see our project website [here](https://alany1.github.io/audio/index.html) and the main block of the `integrate.py`.