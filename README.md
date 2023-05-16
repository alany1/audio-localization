# CLAP: CLIP Localization in Audio-Visual Pairings

This repository contains the code implementation for our project [CLAP: CLIP Localization in Audio-Visual Pairings](https://alany1.github.io/audio/index.html).

<img src="assets/demos.gif" max-width="100%" height="auto" />

## Contents
1. [Setup](#setup)
2. [Extracting Audio and Text Features](#features)
3. [Extracting Video Features](#video)
4. [Running the Model](#integrate)
5. [Examples](#examples)

## Setup <a name="setup"></a>
Install dependencies in a new conda environment:
```
conda create -n clap python=3.7
conda activate clap
pip install -r requirements.txt
```

To run the model, you will need to download the state dictionary of CLIP, AudioCLIP, and other related files. You can download the pre-trained model from [here](https://github.com/AndreyGuzhov/AudioCLIP). Place the model in the `scripts/AudioCLIP/assets` folder. 

## Extracting Audio and Text Features <a name="features"></a>
