import os
import numpy as np
from glob import glob
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data

from . import transforms as my_tf
from myutils import load_image_in_PIL as load_img


def load_image_in_PIL(path, mode='RGB'):
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img.convert(mode)


class WaterDataset(data.Dataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None, eval_size=None):

        super(WaterDataset, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.test_case = test_case
        self.img_list = []
        self.label_list = []
        self.verbose_flag = False
        self.online_augmentation_per_epoch = 640
        self.eval_size = eval_size

        if mode == 'train_offline':
            with open(os.path.join(dataset_path, 'train_imgs.txt')) as f:
                water_subdirs = f.readlines()
            water_subdirs = [x.strip() for x in water_subdirs]

            print('Initialize offline training dataset:')

            for sub_folder in water_subdirs:
                label_list = glob(os.path.join(dataset_path, 'Annotations/', sub_folder, '*.png'))
                label_list.sort(key=lambda x: (len(x), x))
                self.label_list += label_list

                name_list = [os.path.basename(x)[:-4] for x in label_list]

                img_list = glob(os.path.join(dataset_path, 'JPEGImages/', sub_folder, '*.jpg'))
                img_list.sort(key=lambda x: (len(x), x))
                img_list_valid = []
                for img_path in img_list:
                    if os.path.basename(img_path)[:-4] in name_list:
                        img_list_valid.append(img_path)

                self.img_list += img_list_valid

                print('Add', sub_folder, len(img_list_valid), 'files.')



        elif mode == 'eval':
            if test_case is None:
                raise ('test_case can not be None.')

            img_path = os.path.join(dataset_path, 'JPEGImages/', test_case)
            img_list = os.listdir(img_path)
            img_list.sort(key=lambda x: (len(x), x))
            self.img_list = [os.path.join(img_path, name) for name in img_list]

            first_frame_label_path = os.path.join(dataset_path, 'Annotations/', test_case, img_list[0])

            # Detect label image format: png or jpg
            first_frame_label_path = first_frame_label_path[:-3]
            if os.path.exists(first_frame_label_path + 'png'):
                first_frame_label_path += 'png'
            else:
                first_frame_label_path += 'jpg'

            if not os.path.exists(first_frame_label_path):
                label_list = glob(os.path.join(dataset_path, 'Annotations/', test_case, '*.png'))
                label_list.sort(key=lambda x: (x, len(x)))
                first_frame_label_path = label_list[0]

            self.first_frame = load_image_in_PIL(self.img_list[0], 'RGB')
            self.img_list.pop(0)

            self.first_frame_label = load_image_in_PIL(first_frame_label_path, 'P')

            if self.eval_size:
                self.origin_size = self.first_frame.size
                self.first_frame = self.first_frame.resize(self.eval_size, Image.ANTIALIAS)
                self.first_frame_label = self.first_frame_label.resize(self.eval_size, Image.ANTIALIAS)

        else:
            raise ('Mode %s does not support in [train_offline, train_online, eval].' % mode)

    def __len__(self):
        if self.mode == 'train_online':
            return self.online_augmentation_per_epoch
        else:
            return len(self.img_list)

    def get_first_frame(self):
        img_tf = TF.to_tensor(self.first_frame)
        img_tf = my_tf.imagenet_normalization(img_tf)
        return img_tf

    def get_first_frame_label(self):
        return TF.to_tensor(self.first_frame_label)

    def __getitem__(self, index):
        raise NotImplementedError


class WaterDataset_RGB(WaterDataset):
    def __init__(self, mode, dataset_path, input_size=None, test_case=None, eval_size=None):
        super(WaterDataset_RGB, self).__init__(mode, dataset_path, input_size, test_case, eval_size)

    def __getitem__(self, index):
        if self.mode == 'train_offline' or self.mode == 'val_offline' or self.mode == 'test_offline':
            img = load_img(self.img_list[index], 'RGB')
            label = load_img(self.label_list[index], 'P')
            return self.apply_transforms(img, label)
        elif self.mode == 'train_online':
            return self.apply_transforms(self.first_frame, self.first_frame_label)
        elif self.mode == 'eval':
            img = load_img(self.img_list[index], 'RGB')
            if self.eval_size:
                img = img.resize(self.eval_size, Image.ANTIALIAS)
            return self.apply_transforms(img)
        else:
            raise Exception("Error: Invalid dataset mode!")

    def resize_to_origin(self, img):
        return img.resize(self.origin_size)

    def apply_transforms(self, img, label=None):
        if self.mode == 'train_offline' or self.mode == 'train_online':
            img = my_tf.random_adjust_color(img, self.verbose_flag)
            img, label = my_tf.random_affine_transformation(img, None, label, self.verbose_flag)
            img, label = my_tf.random_resized_crop(img, None, label, self.input_size, self.verbose_flag)
        elif self.mode == 'test_offline' or self.mode == 'val_offline':
            img = TF.resize(img, self.input_size)
            label = TF.resize(label, self.input_size)
        elif self.mode == 'eval':
            pass

        img_orig = TF.to_tensor(img)
        img_norm = my_tf.imagenet_normalization(img_orig)

        if self.mode == 'train_offline' or self.mode == 'train_online':
            # label = TF.to_tensor(label)
            label = np.expand_dims(np.array(label, np.float32), axis=0)
            return img_norm, label
        elif self.mode == 'val_offline':
            label = np.expand_dims(np.array(label, np.float32), axis=0)
            return img_norm, label
        elif self.mode == 'test_offline':
            label = np.expand_dims(np.array(label, np.float32), axis=0)
            return img_norm, label, img_orig
        else:
            return None