# V-FloodNet

This is an official PyTorch implementation for paper "V-FloodNet: A Video Segmentation System for Urban Flood Detection and Quantification".

## Environments
We developed and tested the source code under Ubuntu 18.04 and PyTorch framework. 
The following packages are required to run the code.

First, a python virtual environment is recommended. 
I use `pip` to create a virtual environment named `env` and activate it.
Then, recursively pull the submodules code.

```shell
python3 -m venv env
source env/bin/activate
git submodule update --init --recursive
```

In the virtual environment, install the following required packages from their official instructions.

- torch, torchvision, from [PyTorch](https://pytorch.org). We used v1.8.2+cu111 is used in our code. 
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) for scatter operations. We used the version for torch 1.8.1+cu111. 
- [Detectron2](https://github.com/facebookresearch/detectron2) for reference objects segmentation.
- [MeshTransformer](https://github.com/microsoft/MeshTransformer) for human detection and 3D mesh alignment.

We provide the corresponding installation command here

```shell
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html -v
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
cd MeshTransformer
python setup.py build develop
pip install ./manopth/.
```

Then install the rest packages indicated in `requirements.txt`
```shell
pip install -r requirements.txt
```

## Usage

Download and extract the pretrained weights, and put them in the folder `./records/`. Weights and groundtruths are stored in [Google Drive](https://drive.google.com/file/d/1r0YmT24t4uMwi4xtSLXD5jyaMIuMzorS/view?usp=sharing).

### Water Image Segmentation
Put the testing images in `image_folder`, then
```shell
python test_image_seg.py \
    --test_path=/path/to/image_folder --test_name=<test_name>
```
The default output folder is `output/segs/`

### Water Video Segmentation
If your input is a video, we provide a script `scripts/cvt_video_to_imgs.py` to extract frames of the video.
Put the extracted frames in `frame_folder`, then
```shell
python test_video_seg.py \
    --test-path=/path/to/frame_folder --test-name=<test_name>
```

### Water Depth Estimation

We provide three options `stopsign`, `people`, and `ref` for `--opt` to specify three types reference objects.
```shell
python est_waterlevel.py \
  --opt=<opt> --test-name=<test_name> --img-dir=/path/to/img_folder
```
For input video, to compare the estimated water level with the groundtruths in `records/groundtruth/`, you can use 
```shell
python cmp_hydrograph.py --test-name=<test_name>
```

## Copyright
This paper is submitted to Elsevier Journal Computers, Environment and Urban Systems under review. The corresponding author is Xin Li (Xin Li <xin.shane.li@ieee.org>).  All rights are reserved.
