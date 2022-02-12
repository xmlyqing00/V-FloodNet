import argparse
from glob import glob
import os
import torch

from test_video_seg import main


def get_args():
    parser = argparse.ArgumentParser(description='Test Video Segmentation Benchmark')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU card id.')
    parser.add_argument('--budget', type=int, default='250000',
                        help='Max number of features that feature bank can store. Default: 300000')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize data.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the checkpoint (default: none)')
    parser.add_argument('--update-rate', type=float, default=0.1,
                        help='Update Rate. Impact of merging new features.')
    parser.add_argument('--merge-thres', type=float, default=0.95,
                        help='Merging Rate. If similarity higher than this, then merge, else append.')
    parser.add_argument('--benchmark_path', type=str, required=True,
                        help='Benchmark Path')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
    else:
        raise ValueError('CUDA is required. --gpu must be >= 0.')

    assert os.path.isdir(args.benchmark_path)

    palette = [0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 0, 0]

    test_list = glob(os.path.join(args.benchmark_path, '*/'))
    for test_path in test_list:
        test_name = test_path.split('/')[-2]
        args.test_name = test_name
        args.test_path = test_path

        print('Process video', test_name, 'from path', test_path)
        main(args, device, palette)
