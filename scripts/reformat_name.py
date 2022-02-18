import os
from glob import glob
from datetime import datetime
from dateutil import parser



def convert_LSU_creek1(img_dir):

    img_list = glob(os.path.join(img_dir, '*.jpg'))

    for i in range(len(img_list)):

        img_name = os.path.basename(img_list[i])
        # img_time = parser.parse(img_name[:-4])
        img_time = datetime.strptime('2020-04-23-' + img_name[:-4], '%Y-%m-%d-%H_%M')
        time_str = datetime.strftime(img_time, '%Y-%m-%d-%H-%M-%S')
        dst_path = os.path.join(img_dir, time_str + '.jpg')


        cmd_str = f'mv "{img_list[i]}" "{dst_path}"'
        # print(cmd_str)
        os.system(cmd_str)


def convert_LSU_creek2(img_dir):

    img_list = glob(os.path.join(img_dir, '*.jpg'))

    for i in range(len(img_list)):

        img_name = os.path.basename(img_list[i])
        # img_time = parser.parse(img_name[:-4])
        img_time = datetime.strptime(img_name[:-4], '%Y-%m-%d-%H-%M-%S')
        time_str = datetime.strftime(img_time, '%Y-%m-%d-%H-%M-%S')
        dst_path = os.path.join(img_dir, time_str + '.jpg')


        cmd_str = f'mv "{img_list[i]}" "{dst_path}"'
        # print(cmd_str)
        os.system(cmd_str)


def convert_boston_harbor(img_dir):
    img_list = glob(os.path.join(img_dir, '*.jpg'))

    for i in range(len(img_list)):
        img_name = os.path.basename(img_list[i])
        # img_time = parser.parse(img_name[:-4])

        # img_time = datetime.strptime(img_time, '%Y-%m-%d-%H-%M-%S')

        # time_str = datetime.strftime(img_time, '%Y-%m-%d-%H-%M-%S')
        dst_path = os.path.join(img_dir, img_name[:-4] + '.png')
        cmd_str = f'mv "{img_list[i]}" "{dst_path}"'
        print(cmd_str)
        os.system(cmd_str)

if __name__ == '__main__':

    # img_dir = '/Ship01/Dataset/VOS/water/JPEGImages/LSU-20200423'
    img_dir = '/Ship03/Sources/VOS/WaterNetV2/output/AFB-URR_Water_fulltrain/boston_harbor_20190119_20190123_day_s_label_1'

    convert_boston_harbor(img_dir)
