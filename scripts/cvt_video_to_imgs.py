import os
import numpy as np
import cv2

def cvt_video_series_to_images(video_path, out_frames_folder):
    
    if not os.path.exists(out_frames_folder):
        os.makedirs(out_frames_folder)

    video = cv2.VideoCapture(video_path)
    cnt = 0
    stride = 3

    while (video.isOpened()):

        ret, frame = video.read()
        if not ret:
            break

        if cnt % stride == 0:
            out_path = os.path.join(out_frames_folder, f'{cnt:05}.jpg')
            cv2.imwrite(out_path, frame)
        
        cnt += 1

    print('Frame cnt', cnt)
            


if __name__ == '__main__':

    root_folder = '/Ship01/Dataset/VOS/longvideo/Videos'
    out_folder = '/Ship01/Dataset/VOS/longvideo/JPEGImages'
    video_list = os.listdir(root_folder)
    
    for video_name in video_list:

        video_path = os.path.join(root_folder, video_name)
        out_frames_folder = os.path.join(out_folder, video_name[:-4])

        print('Video series path:', video_name)
        print('Out frames folder:', out_frames_folder)
        cvt_video_series_to_images(video_path, out_frames_folder)
