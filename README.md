# Dream Your Room: Multi-Modal Texture Generation and 3D Scene Reconstruction

## Introduction

We propose a system, Dream Your Room. The target is to colorize your room to the indicated style starting from a room tour video. You can feed the system with Text and Image as the style description to specify the style of the room.

We summarize our contributions as follows:
1. We successfully adapted Scentex to handle multimodal inputs, allowing it to process both text and image data. One noteworthy point of our method is the proposed SEE module, which enables the texture generator to "see" the image.
2. Our method demonstrated it is general to real-world scenarios.


For further information of this project, please also check out our 
[slides](https://docs.google.com/presentation/d/14A82o6Mwug3DDvPUK2WIOwWNPNpti8ZJ6PWi2dua5Jc/edit?usp=sharing)
[video](https://youtu.be/d7lcYxTi6v4)

## Setup

The code is tested on Ubuntu 22.04 LTS with PyTorch 2.1.0 CUDA 12.1 installed. Please follow the following steps to install PyTorch first. To run our method, you should at least have a NVIDIA GPU with 24 GB RAM.

```shell
# create and activate the conda environment
conda create -n dyr python=3.9
conda activate dyr

# install PyTorch 2.0.1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then, install PyTorch3D:

```shell
# install runtime dependencies for PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# install PyTorch3D
conda install pytorch3d -c pytorch3d
```

Install `xformers` to accelerate transformers:

```shell
conda install xformers -c xformers
```

Install tinycudann:

```shell
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install the necessary packages listed out in requirements.txt:

```shell
pip install -r requirements.txt
```

Follow the setup process for [COLMAP](https://github.com/colmap/colmap).

## Run the system

For the Scene Reconstruction process, please follow the official page of COLMAP to get your real-world meshes.

For the Texture Generation process:
```shell
./bash/room_1_1_img_learn.sh
```

All generated assets should be found under `outputs`. To configure the style or the target scene, you can modify bash script.


## Acknowledgement

We would like to thank [daveredrum/SceneTex](https://github.com/daveredrum/SceneTex) for providing such a great and powerful codebase for our base model.
