import argparse

from models import cal_camera, cal_dlp


def get_args_parser():
    parser = argparse.ArgumentParser('Set calibration', add_help=False)
    parser.add_argument('--img_path', default='CirclePictures', type=str,
                        help="Path to images for calibration")
    parser.add_argument('--use_extracted_corners', default=True, type=bool,
                        help="Use extracted corners or not")
    parser.add_argument('--corner_folder_path', default='datasets/Corners', type=str,
                        help="Corner folder path")

    # Camera image size
    parser.add_argument('--camera_img_width', default=4096, type=int, help="Width of camera image")
    parser.add_argument('--camera_img_height', default=3072, type=int, help="Height of camera image")
    # Camera pixel size
    parser.add_argument('--du', default=5.5e-3, type=float, help="Width of camera pixel")
    parser.add_argument('--dv', default=5.5e-3, type=float, help="Height of camera pixel")
    # DLP image size
    parser.add_argument('--dlp_img_width', default=912, type=int, help="Width of DLP image")
    parser.add_argument('--dlp_img_height', default=1140, type=int, help="Height of DLP image")

    # L_M parameters
    parser.add_argument('--num_max_iter', default=100, type=int, help="Max Iteration number of L_M")
    parser.add_argument('--tao', default=1e-6, type=float)
    parser.add_argument('--e1', default=1e-5, type=float)
    parser.add_argument('--e2', default=1e-20, type=float)

    return parser


def main(args):
    camera_params = cal_camera(args)
    dlp_params = cal_dlp(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Camera and DLP calibration', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
