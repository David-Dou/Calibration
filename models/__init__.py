from .camera_calibration import calibrate_camera
from .dlp_calibration import calibrate_dlp


def cal_camera(args):
    return calibrate_camera(args)


def cal_dlp(args):
    return calibrate_dlp(args)
