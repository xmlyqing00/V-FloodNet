# WaterNetV2

We build a water level estimation system. It first segments the water regions from each frame/image, then uses the reference objects to estimate the water level.

## Run the image segmentation
```bash
python3 test_image_seg.py \
    --test_path=/path/to/image_folder \
    --test_name=<test_name>
```

## Estimate Water Level / Water Depth

### Stopsign

```bash
python3 est_waterlevel.py \
    --test_name=stopsign \
    --img_dir=/path/to/image_folder \
    --water_mask_dir=./output/test_image_seg/<test_name>/mask
    --opt=stopsign
```

### Skeleton

```bash
python3 est_waterlevel.py \
    --test_name=skeleton \
    --img_dir=/path/to/image_folder \
    --water_mask_dir=./output/test_image_seg/<test_name>/mask
    --opt=skeleton
```
https://tidesandcurrents.noaa.gov/waterlevels.html?id=8443970&type=Tide+Data&name=Boston&state=MA

<!-- 
In the first part, we have two modules to segment the water region, the first one is the video module, and the second one is the image module.
- Video module: Take the first frame annotation as input, segment the water masks from the rest frames.
- Image module: Use the prior water image dataset, it can automatically segment the water region and reference objects from image.

As for the second part, water level estimation. It has several options depend on the reference object.
- Reference object is fixed (in `est_waterlevel.py`)
- Reference object is selected by user (in `est_waterlevel.py`)
- Reference object is either human skeleton or stop sign (in `seg_image.py`) -->
  
<!-- To do list
- [] Get familiar with PyTorch
- [ ] Run codes: Segment the water region by video module (the pretrained model is available on Google drive https://drive.google.com/drive/folders/1sU5rTSotwR1e3bmlH8Ux_x4gTBQb-QvO?usp=sharing)  
- [ ] Run codes: Estimate the water level by user selected references. In `est_waterlevel.py`
- [ ] Run codes: Segment the water region by image module.
- [ ] Run codes: Estimate the water level by the skeleton and stop sign.
- [ ] Think about the data representations, how do we make it together? how do we store water segmentation results and reference object information? How can we combine the image-based segmentation and the video-based segmentation together?
- [ ] Reorganize the code structure.
- [ ] Reorganize the water dataset 
- [ ] Run the experiments.
- [ ] Compare other methods with ours.


[comment]: <> (--config-file)

[comment]: <> (../projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml)

[comment]: <> (--input)

[comment]: <> (/Ship01/Dataset/VOS/water/JPEGImages/test_imgs/8.jpg)

[comment]: <> (--output)

[comment]: <> (/Ship01/tmp/human/) -->
