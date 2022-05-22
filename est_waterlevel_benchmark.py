import argparse
import os
from glob import glob

from est_waterlevel import main


def get_parser():
    parser = argparse.ArgumentParser(description='Estimate Water Level')
    parser.add_argument('--water-mask-dir-prefix', type=str, default='./output', required=True,
                        help='Path to the water mask folder.')
    parser.add_argument('--out-dir', default='output/waterlevel',
                        help='A file or directory to save output results.')
    parser.add_argument('--opt', type=str,
                        help='Estimation options.')
    parser.add_argument('--benchmark-path', type=str, required=True,
                        help='Benchmark Path')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_parser()
    print(args)

    test_list = sorted(glob(os.path.join(args.benchmark_path, '*/')))
    for test_path in test_list:
        args.img_dir = test_path
        test_name = test_path.split('/')[-2]
        args.test_name = test_name
        args.water_mask_dir = os.path.join(args.water_mask_dir_prefix, test_name, 'mask')

        print('Process video', test_name, 'from path', test_path)
        main(args)
