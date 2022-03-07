import argparse
import os
from glob import glob
import torch

from estimation.object_detection import est_by_obj_detection
from estimation.reference_tracking import est_by_reference


def get_parser():
    parser = argparse.ArgumentParser(description='Estimate Water Level')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU card id.')
    parser.add_argument('--test-name', type=str, required=True,
                        help='Name of the test video')
    parser.add_argument('--water-mask-dir', type=str, default='./output',
                        help='Path to the water mask folder.')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='Input image directory.')
    parser.add_argument('--out-dir', default='output/waterlevel',
                        help='A file or directory to save output results.')
    parser.add_argument('--opt', type=str,
                        help='Estimation options.')

    return parser.parse_args()


def main(args):

    # if args.gpu >= 0 and torch.cuda.is_available():
    #     device = torch.device('cuda', args.gpu)
    # else:
    #     device = torch.device('cpu')

    img_list = sorted(glob(os.path.join(args.img_dir, '*.jpg')) + glob(os.path.join(args.img_dir, '*.png')))
    water_mask_list = sorted(glob(os.path.join(args.water_mask_dir, '*.png')))
    out_dir = os.path.join(args.out_dir, f'{args.test_name}_{args.opt}')
    os.makedirs(out_dir, exist_ok=True)

    if args.opt in ['skeleton', 'stopsign']:
        est_by_obj_detection(img_list, water_mask_list, out_dir, args.opt)
    elif args.opt == 'ref':
        if 'houston' in args.test_name:
            enable_tracker = False
            enable_calib = False
            tracker_num = 2
        elif 'LSU' in args.test_name:
            enable_tracker = False
            enable_calib = False
            tracker_num = 1
        elif 'boston' in args.test_name:
            enable_tracker = True
            enable_calib = True
            tracker_num = 2

        est_by_reference(img_list, water_mask_list, out_dir, tracker_num, enable_tracker, enable_calib)
    else:
        raise NotImplementedError(args.opt)


if __name__ == '__main__':

    _args = get_parser()
    print(_args)

    main(_args)
    exit(0)
    #
    # if args.img_dir[-1] == '/':
    #     args.img_dir = args.img_dir[:-1]
    # img_dir_name = os.path.basename(args.img_dir)
    # seg_dir = os.path.join(args.out_dir, img_dir_name, 'seg')
    # viz_dir = os.path.join(args.out_dir, img_dir_name, 'viz')
    #
    # if not os.path.exists(seg_dir):
    #     os.makedirs(seg_dir)
    # if not os.path.exists(viz_dir):
    #     os.makedirs(viz_dir)
    #
    # img_segmentation = ImageBasedModel(user_config, device, args.water_path)
    # waterdepth_esitmation = myutils.WaterdepthEstimation(img_segmentation.metadata)
    #
    # img_list = sorted(glob.glob(os.path.join(args.img_dir, '*.jpg')) + glob.glob(os.path.join(args.img_dir, '*.png')))
    #
    # for path in tqdm.tqdm(img_list):
    #     # use PIL, to be consistent with evaluation
    #     img = read_image(path, format='BGR')
    #
    #     img = myutils.resize_img(img, 1280)
    #
    #     seg_res, viz_dict = img_segmentation.seg_img(img)
    #
    #     img_name = os.path.basename(path)[:-4]
    #     seg_res_path = os.path.join(seg_dir, img_name + '_seg_by_img.pkl')
    #     with open(seg_res_path, 'wb') as f:
    #         pickle.dump(seg_res, f)
    #
    #     img_water_path = os.path.join(viz_dir, img_name + '_water.png')
    #     cv2.imwrite(img_water_path, seg_res['water_mask'])
    #
    #     viz_img_water_path = os.path.join(viz_dir, img_name + '_water_by_img.png')
    #     cv2.imwrite(viz_img_water_path, viz_dict['water_by_img'])
    #     viz_img_water_path = os.path.join(viz_dir, img_name + '_seg_by_img.png')
    #     cv2.imwrite(viz_img_water_path, viz_dict['seg_by_img'])
    #
    #     # continue
    #
    #     water_depth, viz_dict = waterdepth_esitmation.est(seg_res, viz_dict, img)
    #     viz = myutils.Visualizer(img, viz_dir, img_name)
    #     if 'skeleton' in water_depth:
    #         viz.plot_depth(water_depth['skeleton'], water_depth['skeleton_vlist'], seg_res['water_mask'], suffix='skeleton')
    #     if 'stopsign' in water_depth:
    #         viz.plot_depth(water_depth['stopsign'], water_depth['stopsign_vlist'], seg_res['water_mask'], suffix='stopsign')
    #
    #     continue
    #     basename = os.path.basename(path)[:-4]
    #     water_mask_path = os.path.join(args.output, basename + '_water_mask.npy')
    #     water_mask = np.load(water_mask_path)
    #
    #     water_viz = np.concatenate((water_mask * 200,
    #                                 np.zeros_like(water_mask),
    #                                 np.zeros_like(water_mask)), axis=2).astype(np.uint8)
    #
    #
    #     if args.output:
    #         # if os.path.isdir(args.output):
    #         #     assert os.path.isdir(args.output), args.output
    #         #     out_filename = os.path.join(args.output, os.path.basename(path))
    #         # else:
    #         #     assert len(args.input) == 1, 'Please specify a directory with args.output'
    #         #     out_filename = args.output
    #
    #         out_filename = os.path.join(args.output, basename + '_skeleton.jpg')
    #         visualized_output.save(out_filename)
    #
    #         viz_img = visualized_output.img + water_mask
    #         out_filename = os.path.join(args.output, basename + '_all.jpg')
    #         cv2.imwrite(out_filename, viz_img)
    #
    #         # water_img = (water_img * 255).astype(np.uint8)
    #         # water_mask = (water_mask * 255).astype(np.uint8)
    #         # cv2.imwrite(, water_img)
    #         # cv2.imwrite(os.path.join(args.output, basename + '_water_mask.jpg'), water_mask)
    #     else:
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    #         cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    #         cv2.imshow('Water Image', water_img)
    #         cv2.imshow('Water Mask', water_mask)
    #         if cv2.waitKey(0) == 27:
    #             break  # esc to quit
    #
    # elif args.video_input:
    #     video = cv2.VideoCapture(args.video_input)
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frames_per_second = video.get(cv2.CAP_PROP_FPS)
    #     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     basename = os.path.basename(args.video_input)
    #
    #     if args.output:
    #         if os.path.isdir(args.output):
    #             output_fname = os.path.join(args.output, basename)
    #             output_fname = os.path.splitext(output_fname)[0] + '.mkv'
    #         else:
    #             output_fname = args.output
    #         assert not os.path.isfile(output_fname), output_fname
    #         output_file = cv2.VideoWriter(
    #             filename=output_fname,
    #             # some installation of opencv may not support x264 (due to its license),
    #             # you can try other format (e.g. MPEG)
    #             fourcc=cv2.VideoWriter_fourcc(*'x264'),
    #             fps=float(frames_per_second),
    #             frameSize=(width, height),
    #             isColor=True,
    #         )
    #     assert os.path.isfile(args.video_input)
    #     for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
    #         if args.output:
    #             output_file.write(vis_frame)
    #         else:
    #             cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
    #             cv2.imshow(basename, vis_frame)
    #             if cv2.waitKey(1) == 27:
    #                 break  # esc to quit
    #     video.release()
    #     if args.output:
    #         output_file.release()
    #     else:
    #         cv2.destroyAllWindows()
