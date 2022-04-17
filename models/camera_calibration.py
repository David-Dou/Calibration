import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
from utils import homography as Homo


class CameraCalibration:
    def __init__(self, width, height, du, dv, num_max_iter, tao, e1, e2,
                 use_extracted_corners=False, corner_folder_path=None):
        self.use_extracted_corners = use_extracted_corners
        if corner_folder_path is None and use_extracted_corners is True:
            raise ValueError("Corner path should be provided")
        self.corner_folder_path = corner_folder_path

        self.width = width
        self.height = height
        self.du = du
        self.dv = dv
        self.num_iter = num_max_iter
        self.tao = tao
        self.e1 = e1
        self.e2 = e2

        self.imgpoints_list = []
        self.objpoints_list = []

        self.H_list = []

        self.A = np.eye(3, dtype=np.float64)
        self.A[0][2], self.A[1][2] = self.width / 2, self.height / 2

    def get_corners(self):
        print("\nStart getting corners...")

        if self.use_extracted_corners:
            '''
            Circle corners reading
            '''
            corner_path_list = glob(os.path.join(self.corner_folder_path, "*.csv"))

            for i, corner_path in enumerate(corner_path_list):
                corner = pd.read_csv(corner_path)
                imgps_x_one_img, imgps_y_one_img = corner.imgps_x.values, corner.imgps_y.values
                # imgps_x_one_img, imgps_y_one_img = corner["imgps_x"], corner["imgps_y"]
                objps_x_one_img, objps_y_one_img = corner.objps_x.values, corner.objps_y.values

                imgpoints = np.concatenate((imgps_x_one_img, imgps_y_one_img)).reshape(2, -1)

                objpoints = np.concatenate((objps_x_one_img, objps_y_one_img)).reshape(2, -1)

                self.imgpoints_list.append(imgpoints)
                self.objpoints_list.append(objpoints)

    def get_in_params(self):
        self.get_corners()

        assert len(self.imgpoints_list) >= 5, "Number of valid photos isn't enough"

        print("\nStart calculating intrinsic parameters...")

        G = np.zeros(shape=(len(self.imgpoints_list), 5), dtype=np.float64)

        for img_idx, (objps, imgps) in enumerate(zip(self.objpoints_list, self.imgpoints_list)):
            objps_h, imgps_h = Homo.get_h_coord(objps), Homo.get_h_coord(imgps)
            '''
            How to check by cv.findhomography?
            '''
            H = Homo.h_for_bite_ca(objps_h, imgps_h)
            self.H_list.append(H)

            G[img_idx][0] = H[0][0] ** 2 + H[0][1] ** 2
            G[img_idx][1] = H[1][0] ** 2 + H[1][1] ** 2
            G[img_idx][2] = 2 * H[0][0] * H[1][0] + 2 * H[0][1] * H[1][1]
            G[img_idx][3] = H[1][0] ** 2 + H[1][1] ** 2
            G[img_idx][4] = -(H[0][0] * H[1][1] - H[0][1] * H[1][0]) ** 2

        b = np.dot(np.linalg.pinv(G), np.ones(shape=(len(self.imgpoints_list),), dtype=np.float64))

        m = self.du / np.sqrt(b[0])
        print("m_init =", m)

        theta = np.arctan2(np.sqrt(b[3] / b[0]) * self.du / self.dv, b[2] / b[0])
        print("theta_init =", 180 * theta / np.pi)

        self.A[0][0] = m / self.du
        self.A[0][1] = - m / (self.dv * np.tan(theta))
        self.A[1][1] = m / (self.dv * np.sin(theta))
        print("intrinsic matrix A_init =\n", self.A)

        return {
            "m": m,
            "theta": theta,
            "intrinsic matrix": self.A
        }

    def get_ex_params(self):
        print("\nStart calculating Ks...")

        ex_param_list = []

        for H in self.H_list:
            Ks = np.dot(np.linalg.inv(self.A), H)
            ex_rotate_param = Ks[:2, :2]

            padding_vec = np.zeros(shape=(2,), dtype=np.float32)
            if ex_rotate_param[0][0] ** 2 + ex_rotate_param[0][1] ** 2 < 1:
                padding_vec[0] = (1 - ex_rotate_param[0][0] ** 2 - ex_rotate_param[0][1] ** 2) ** 0.5
            if ex_rotate_param[1][0] ** 2 + ex_rotate_param[1][1] ** 2 < 1:
                padding_vec[1] = (1 - ex_rotate_param[1][0] ** 2 - ex_rotate_param[1][1] ** 2) ** 0.5

            padding_vec = np.array([padding_vec])

            ex_rotate_param = np.concatenate((ex_rotate_param, padding_vec.T), axis=1)
            ex_rotate_param = np.concatenate((ex_rotate_param, np.zeros(shape=(1, ex_rotate_param.shape[1]))), axis=0,
                                             dtype=np.float32)

            ex_transform_param = np.array([Ks[:, 2]])

            ex_param = np.concatenate((ex_rotate_param, ex_transform_param.T), axis=1)
            ex_param_list.append(ex_param)

        return ex_param_list
    
    


def calibrate_camera(args):
    camera_calibration = CameraCalibration(
        args.camera_img_width,
        args.camera_img_height,
        args.du,
        args.dv,
        args.num_max_iter,
        args.tao,
        args.e1,
        args.e2,
        args.use_extracted_corners,
        args.corner_folder_path
    )

    camera_params = {
        "intrinsic params": camera_calibration.get_in_params(),
        "extrinsic params": camera_calibration.get_ex_params()
    }

    with open("results/camera_params.pkl", "wb") as f:
        pickle.dump(camera_params, f)

    return camera_params
