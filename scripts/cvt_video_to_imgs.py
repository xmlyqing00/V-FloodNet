import os
import numpy as np
import cv2
from glob import glob

def cvt_video_series_to_images(video_path, out_frames_dir):
    
    os.makedirs(out_frames_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    cnt = 0
    stride = 3

    while (video.isOpened()):

        ret, frame = video.read()
        if not ret:
            break

        if cnt % stride == 0:
            out_path = os.path.join(out_frames_dir, f'{cnt:05}.jpg')
            cv2.imwrite(out_path, frame)
        
        cnt += 1

    print('Frame cnt', cnt)
            


if __name__ == '__main__':

    in_dir = '/home/gvc/Datasets/water2/University Lakes'
    out_dir = '/home/gvc/Datasets/water2/test_university_lakes'
    video_list = glob(os.path.join(in_dir, '*.MOV'))
    
    for video_path in video_list:

        video_name = os.path.basename(video_path)[:-4]
        out_frames_dir = os.path.join(out_dir, video_name)

        print('Video series path:', video_name)
        print('Out frames folder:', out_frames_dir)
        cvt_video_series_to_images(video_path, out_frames_dir)
