# V-FloodNet

This is an official PyTorch implementation for paper "V-FloodNet: A Video Segmentation System for Urban Flood Detection and Quantification". A robust automatic system for water level or inundation depth estimation from images and videos, consisting of reliable
water and reference object detection/segmentation, and depth estimation models. 

This paper was published in Environmental Modelling & Software, Elsevier (2023). Click [here](https://www.sciencedirect.com/science/article/pii/S1364815222002869?casa_token=-eyHhrbYSpAAAAAA:jddGSA5W9ME5YMJbMiMoZ5UfBf3qrQbEyhLWTBP7akl9Pq8tnHXxzVJBFb9D0PDlmREeaoWnJAc) to view the paper.

Here are some screenshots of our results. We can estimate water depth from the detected objects in the scene.
![](assets/screenshot_people.png)

We can also estimate water level from from long videos under various weather and illumination conditions.
![](https://res.cloudinary.com/xmlyqing00/image/upload/v1734297629/vfloodnet-demo-800x600_w8cwkb.gif)
![](assets/screenshot_video.png)

We provide our **dataset** and **results** in [Google Drive](https://drive.google.com/drive/folders/1DURwcb_qhBeWYznTrpJ-7yGJTHxm7pxC?usp=sharing).
More demo videos are available in the `animation_videos` folder.

## 1 Environments

### 1.1 Code and packages
We developed and tested the source code on Ubuntu 18.04 and PyTorch framework. Nvidia GPU with driver is required.

For Windows users, running our code on WSL2 subsystem is recommended. Because part of our code has GUI window, you need to install X Server on Windows. Please refer to [this post](https://techcommunity.microsoft.com/t5/windows-dev-appconsult/running-wsl-gui-apps-on-windows-10/ba-p/1493242) for instructions.

**Update for WSL2 users**: 
1. We found that the submodule MeshTransformer contains multiple softlink. When you are using WSL2, make sure you clone the repo into a **NTFS** disk, otherwise the softlink doesn't work.
2. The end-of-line-sequence for bash files should be **LF** instead of CRLF, you can easily change it by setting git config or in Visual Studio Code.

The following packages are required to run the code. First, git clone this repository
```bash
git clone https://github.com/xmlyqing00/V-FloodNet.git
cd V-FloodNet
git submodule update --init --recursive
```

Second, a python virtual environment is recommended. PyTorch 1.8.2 LTS requires the `python<=3.8`. We also provide the installation guidance if you want to use higher version of PyTorch.

```shell
conda create -n vflood python=3.8
conda activate vflood
```

In the virtual environment, install the following required packages from their official instructions.

- torch, torchvision, from [PyTorch](https://pytorch.org). We used v1.8.2+cu111 is used in our code or v2.2.0+cu121. 
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) for scatter operations. We used the version for torch 1.8.1+cu111 or not specified. 
- [Detectron2](https://github.com/facebookresearch/detectron2) for reference objects segmentation. We use version 0.6 here.
- [MeshTransformer](https://github.com/microsoft/MeshTransformer) for human detection and 3D mesh alignment.

<details>
<summary> Installation commands for CUDA 11.1 (NVIDIA GPU older than 40X0 such as 3090) </summary>

```shell
# Install essential libaries
sudo apt install build-essential libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev

# Install PyTorch
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html -v

# Install Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

# Install MeshTransformer, make sure you are in the folder of this repository.
cd MeshTransformer
python setup.py build develop
pip install ./manopth/.
cd ..

# Install other python packages
pip install -r requirements.txt
```

</details>

<details>
<summary> Installation commands for CUDA 12.1 (NVIDIA GPU 40X0) </summary>

Since our package depends on several PyTorch packages which are not updated frequently. Modern GPU cards such as 4090 can not run on the old PyTorch version. We need to use some tricks to install the dependants. I tested the following environment on NVIDIA 4090 and CUDA 12.1. `nvcc` is not required.

First, we need to install the `gh` tool which can switch repository to a pull request version
```shell
sudo apt install gh
gh auth login
```

Second, we use the following commands to install the libraries
```shell
# Install essential libaries
sudo apt install build-essential libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev

# Install nightly PyTorch 2.2.0 which supports the latest CUDA 12.1
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install torch-scatter 

# Install Detectron2, we have to manually compile it instead of using wheel.
cd ..
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2/
gh pr checkout 4868
pip install -e .
cd ../V-FloodNet/

# Install MeshTransformer, make sure you are in the folder of this repository.
cd MeshTransformer

# Manually create the __init__.py file to include the package.
touch metro/modeling/data/__init__.py
touch metro/modeling/hrnet/__init__.py
python setup.py build develop
pip install ./manopth/.
cd ..

# Install other python packages
pip install -r requirements.txt
```

</details>

### 1.2 Pretrained Models

<details>
<summary>
Links to pretrained models and dataset.
</summary>

First, run the following script to download the pretrained models of MeshTransformer (it may take about 15 mins)
```bash
sh scripts/download_MeshTransformer_models.sh
```

Second, due to the license, we have to manually download some body files.
1. Download SMPL model `SMPLIFY_CODE_V2.ZIP` from the official website [SMPLify](http://smplify.is.tue.mpg.de/). When it starts downloading, it shows `mpips_smplify_public_v2.zip`. 
2. Extract it and copy the model file `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from `smplify_public/code/models/` to `V-FloodNet/MeshTransformer/metro/modeling/data/`.

Third, download the archives from [Google Drive](https://drive.google.com/drive/folders/1DURwcb_qhBeWYznTrpJ-7yGJTHxm7pxC?usp=sharing).
1. Extract the pretrained models for water segmentation `records.zip` and put them in the folder `V-FloodNet/records/`. 
2. Extract the water dataset `WaterDataset` in any path, which includes the training images and testing videos. You can put it at `V-FloodNet/WaterDataset/`.

</details>

The final file structure looks like
```
V-FloodNet/
  -- MeshTransformer/
    -- metro/
      -- modeling/
        -- data/
          -- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
  -- records/
    -- groundtruth/
    -- link_efficientb4_model.pth
    -- video_seg_checkpoint_20200212-001734.pth
    ...
  -- WaterDataset/
    -- train_images/
    -- test_videos/
  ...
```


## 2 Usage

### 2.1 Water Image Segmentation
Put the testing images in a folder then
```shell
python test_image_seg.py --test-path=/path/to/image/folder --test-name=<anyname>
```
The default output folder is `output/segs/`

<details>
<summary>
Example
</summary>

```shell
python test_image_seg.py --test-path=assets/img_exp/ --test-name img_exp
```

</details>

### 2.2 Water Video Segmentation
If your input is a video, we provide a script `scripts/cvt_video_to_imgs.py` to extract frames of the video.
Put the extracted frames in a folder then
```shell
python test_video_seg.py --test-path=/path/to/image/folder --test-name=<anyname>
```

<details>
<summary>
Example
</summary>

```shell
python test_video_seg.py --test-path=assets/lake_exp/ --test-name lake_exp
```

</details>


More arguments can be found in `python test_video_seg.py --help`.

Note that the `--budget` argument controls the maximum number of features that can be stored in GPU memory for adaptive feature bank. The default value `3000000` is an experimental number for 11GB GPU memory.

### 2.3 Water Depth Estimation

Arguments for water depth estimation

1. `--test-path` indicates the path to the image forder.
2. `--test-name` here should be **as the same as** the above segmentations.
3. `--opt`: should be one of the three options `stopsign`, `people`, and `ref` to specify three types reference objects. 
```shell
python est_waterlevel.py --test-path=/path/to/image/folder --test-name=<anyname> --opt=<opt>
```


<details>
<summary>
Example
</summary>

```shell
python est_waterlevel.py --test-path=assets/lake_exp/ --test-name lake_exp --opt people
python est_waterlevel.py --test-path=assets/lake_exp/ --test-name lake_exp --opt stopsign
```

</details>


This is an optional step. For input video, to compare the estimated water level with the groundtruths in `records/groundtruth/`, you can use 
```shell
python scripts/cmp_hydrograph.py --test-name=<anyname>
```

## 3 Training

Retrain the model from scratch or your own data could take a long time for weeks. We strongly recommend you used our pretrained model for inferring.
To train the image segmentation module, please refer to the `train_image_seg.py`
```
python train_image_seg.py \
  --dataset-path=/path/to/Datasets/WaterDataset/train_images/ \
  --encoder=efficientnet-b4
```

To train the video segmentation module, please refer to the `train_video_seg.py`. We provide an initial training point in [Google Drive](https://drive.google.com/file/d/1l9TXNV4YQMAxL8RqfL14-qofn_s641Dx/view?usp=sharing), where the network weights are pretrained on large general videos.
We use the following parameters to train our model
```
python train_video_seg.py \
  --dataset=/path/to/WaterDataset/train_images \
  --resume=./records/level2_YouTubeVOS.pth \
  --new --log
```

## 4 Copyright
```
@article{liang2023v,
  title={V-FloodNet: A video segmentation system for urban flood detection and quantification},
  author={Liang, Yongqing and Li, Xin and Tsai, Brian and Chen, Qin and Jafari, Navid},
  journal={Environmental Modelling \& Software},
  volume={160},
  pages={105586},
  year={2023},
  publisher={Elsevier}
}
```
This paper was published in Environmental Modelling & Software, Elsevier (2023). The corresponding author is Xin Li (Xin Li <xin.shane.li@ieee.org>).  All rights are reserved.
