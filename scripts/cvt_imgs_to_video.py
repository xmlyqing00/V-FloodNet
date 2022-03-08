import argparse
import cv2
import numpy as np
import os
from glob import glob
from tqdm import trange



def get_parser():
    parser = argparse.ArgumentParser(description='Convert images to videos.')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='../output/videos',
                        help='Path to the input image')
    parser.add_argument('--video-name', type=str, required=True,
                        help='Name of the test video')
    args = parser.parse_args()
    return args


def cvt_images_to_video(image_folder,
                        video_dir,
                        video_name,
                        video_len,
                        fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
                        stride=1,
                        start=0,
                        fps=10):

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    video_path = os.path.join(video_dir, f'{video_name}.mp4')

    img_list = glob(os.path.join(image_folder, '*.png')) + glob(os.path.join(image_folder, '*.jpg'))
    if len(img_list) == 0:
        exit(-1)
    img_list.sort(key=lambda x: (len(x), x))
    # img_list = img_list[3:]

    if video_len == -1:
        end = len(img_list)
    else:
        end = min(int(start + fps * video_len), len(img_list))
    first_image_path = os.path.join(image_folder, img_list[0])
    first_image = cv2.imread(first_image_path)
    height, width, channels = first_image.shape
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    stride = max(0, int(stride))

    for image_idx in trange(start, end, stride):
        image_path = os.path.join(image_folder, img_list[image_idx])
        image = cv2.imread(image_path)
        video.write(image)

    video.release()

    print(video_path)


if __name__ == '__main__':
    args = get_parser()
    video_len = -1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    stride = 1
    fps = 3

    cvt_images_to_video(args.img_dir, args.out_dir, args.video_name, video_len, fourcc, stride, fps=fps)