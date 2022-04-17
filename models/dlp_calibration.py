import os
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from utils import homography as Homo
from matplotlib import pyplot as plt


class DLPCalibration:
    def __init__(self, width, height, use_extracted_corners=False, corner_folder_path=None):
        self.width = width
        self.height = height
        self.use_extracted_corners = use_extracted_corners
        self.corner_folder_path = corner_folder_path

        self.imgpoints_list = []
        self.objpoints_list = []

    def get_corners(self):
        corner_path_list = glob(os.path.join(self.corner_folder_path, "*.csv"))

        for corner_path in corner_path_list:
            corner = pd.read_csv(corner_path)
            imgps_x_one_img, imgps_y_one_img = corner.u.values, corner.v.values
            objps_x_one_img, objps_y_one_img = corner.objps_x.values, corner.objps_y.values

            imgpoints = np.concatenate((imgps_x_one_img, imgps_y_one_img), dtype=np.float32).reshape(2, -1)
            objpoints = np.concatenate((objps_x_one_img, objps_y_one_img), dtype=np.float32).reshape(2, -1)

            self.imgpoints_list.append(imgpoints)
            self.objpoints_list.append(objpoints)

    def get_params(self):
        self.get_corners()

        assert len(self.imgpoints_list) >= 3, "Number of valid photos isn't enough"

        print("\nStart calculating params of DLP: ")
        # using opencv
        objpoints_list, imgpoints_list = [], []
        for (objpoints_2D, imgpoints) in zip(self.objpoints_list, self.imgpoints_list):
            objpoints_3D = np.concatenate((objpoints_2D, np.zeros(shape=(1, objpoints_2D.shape[1]))), axis=0,
                                          dtype=np.float32)
            objpoints_list.append(objpoints_3D.T)
            imgpoints_list.append(imgpoints.T)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints_list, imgpoints_list, [self.width, self.height],
                                                          None, None,
                                                          flags=cv.CALIB_TILTED_MODEL +
                                                                cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 +
                                                                cv.CALIB_FIX_K3 + cv.CALIB_FIX_K4 +
                                                                cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                                                                cv.CALIB_FIX_S1_S2_S3_S4 + cv.CALIB_ZERO_TANGENT_DIST)

        print('mtx=', mtx)
        print('tauX=', dist[0][12])
        print('tauY=', dist[0][13])

        dlp_ex_params = []
        for rvec, tvec in zip(rvecs, tvecs):
            rmat, _ = cv.Rodrigues(rvec)
            ex_param_mat = np.concatenate((rmat, tvec), axis=1)
            dlp_ex_params.append(ex_param_mat)

        return {
            "tauX": dist[0][12],
            "tauY": dist[0][13],
            "intrinsic matrix": mtx,
            "extrinsic params": dlp_ex_params
        }


def calibrate_dlp(args):
    dlp_calibration = DLPCalibration(
        args.dlp_img_width,
        args.dlp_img_height,
        args.use_extracted_corners,
        args.corner_folder_path
    )

    dlp_params = dlp_calibration.get_params()

    with open("results/dlp_params.pkl", "wb") as f:
        pickle.dump(dlp_params, f)

    return dlp_params
