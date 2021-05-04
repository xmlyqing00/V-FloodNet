import os
import argparse
from glob import glob


def cvt_imgs():
    src_dir = args.dir

    json_files = sorted(glob(os.path.join(src_dir, '*.json')))

    for json_path in json_files:
        cmd = f'labelme_json_to_dataset {json_path}'
        print(cmd)
        os.system(cmd)

        base_name = os.path.basename(json_path)[:-5]
        src_path = os.path.join(src_dir, base_name + '_json', 'label.png')
        dst_path = os.path.join(src_dir, base_name + '.png')
        cmd = f'cp {src_path} {dst_path}'
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch Convert Labelme')
    parser.add_argument(
        '--dir', required=True, type=str, metavar='PATH', help='Path to json dir.')
    args = parser.parse_args()

    cvt_imgs()
