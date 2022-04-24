import pickle
import argparse

from models import cal_camera, cal_dlp
from utils import rebuild

import main as calibration


def parse_args():
    calibration_parser = calibration.get_args_parser()
    parser = argparse.ArgumentParser("Run with rebuild 3D object", parents=[calibration_parser])
    parser.add_argument("--use_calibrated_params", default=True, type=bool,
                        help="Use calibrated params or not")
    parser.add_argument("--calibrated_params_path", default="results", type=str)
    parser.add_argument("--rebuild_file", default="demo/rebuild_demo.csv", type=str)
    parser.add_argument("--vis_color_interval", default=['b', 'g', 'y', 'orangered'], type=list)
    parser.add_argument("--vis_height_interval", default=[[-1, 0.1], [0.3, 1.1], [1.1, 1.8], [1.8, 3]],
                        type=list)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.use_calibrated_params:
        camera_params = cal_camera(args)
        dlp_params = cal_dlp(args)
    with open(args.calibrated_params_path + "/camera_params.pkl", "rb") as f_camera:
        camera_params = pickle.load(f_camera)
    with open(args.calibrated_params_path + "/dlp_params.pkl", "rb") as f_dlp:
        dlp_params = pickle.load(f_dlp)

    camera_mtx = camera_params["intrinsic params"]["intrinsic matrix"]
    camera_ex_param = camera_params["extrinsic params"][0]
    dlp_mtx = dlp_params["intrinsic matrix"]
    dlp_ex_param = dlp_params["extrinsic params"][0]

    rebuild(args.rebuild_file, camera_mtx, camera_ex_param, dlp_mtx, dlp_ex_param,
            args.vis_color_interval, args.vis_height_interval)


if __name__ == '__main__':
    main()
