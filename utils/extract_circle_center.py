import os
import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt


class ExtractCircleCenter:
    def __init__(self, file_path, Nc=3, img_height=3072, img_width=4096):
        self.file_path = file_path
        self.img_height = img_height
        self.img_width = img_width
        self.Nc = Nc

    def img_read(self):
        img_path_list = glob(os.path.join(self.file_path, "*.raw"))  # list of filenames, out of order

        img_list = []
        for fname in img_path_list:
            img_16bit = np.fromfile(fname, dtype=np.uint16).reshape(self.img_height, self.img_width)
            min_16bit, max_16bit = np.min(img_16bit), np.max(img_16bit)
            img_8bit = np.array(np.rint(255 * ((img_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
            # img_name = (fname.split("\\")[-1]).split(".")[0]
            # plt.title("img " + img_name)
            # plt.imshow(img_8bit, cmap='gray')
            # plt.show()
            # '''
            # Show images of right size
            # '''
            # cv.namedWindow('img', cv.WINDOW_FREERATIO)
            # cv.imshow('img', img_8bit)
            # cv.waitKey(0)
            img_list.append(img_8bit)
        '''
        same index to choose the picture and its path
        '''
        return img_list, img_path_list

    def extract_circle_center(self):
        img_list, img_path_list = self.img_read()

        for img_gray, img_path in zip(img_list, img_path_list):
            imgps_x_one_img, imgps_y_one_img = [], []
            long_axis_one_img, short_axis_one_img = [], []

            img_name = (img_path.split("\\")[-1]).split(".")[0]
            # use median filter to denoise
            img_blurred = cv.medianBlur(img_gray, ksize=41)
            # use canny to extract edges
            img_edge = cv.Canny(img_blurred, 40, 100)
            # generate binary images
            ret, img_binary = cv.threshold(img_edge, 127, 255, cv.THRESH_BINARY)
            '''
            cv.findContours:param
                            :return contours: tuple of ndarray(points)
                                        all contours in input image, one element of tuple means points on one contour
            '''
            contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            for cnt in contours:
                num_points = len(cnt)
                num_points_low_threshold, num_points_high_threshold = 100, 1000
                if num_points > num_points_low_threshold and num_points < num_points_high_threshold:
                    valid_contours.append(cnt)
                    '''
                    cv.fitEllipse:param cnt(ndarray):points, usually element of contours from cv.findContours
                                 :return ellipse(tuple):((x, y) , (a, b), angle)
                                                        (x, y):center of ellipse
                                                        (a, b):diameter of the major axis
                                                        angle:rotation angle of center
                    '''
                    ellipse = cv.fitEllipse(cnt)
                    imgps_x_one_img.append(ellipse[0][0])
                    imgps_y_one_img.append(ellipse[0][1])
                    long_axis_one_img.append(ellipse[1][0])
                    short_axis_one_img.append(ellipse[1][1])

                    refined_contours = []
                    for i, contour in enumerate(valid_contours):
                        center = (imgps_x_one_img[i], imgps_y_one_img[i])
                        refined_contours.append(self.refine_coutour(contour, center, img_gray))
                    img_color = cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)
                    img_cnt = cv.drawContours(img_color, refined_contours, -1, (255, 0, 0), 3)
                    plt.imshow(img_cnt)
                    plt.show()

            # measures = np.array(long_axis_one_img) * np.array(short_axis_one_img)
            # del_idx_list = []
            # for point_idx in range(len(measures)):
            #     if measures[point_idx] < np.mean(measures):
            #         del_idx_list.append(point_idx)
            # for counter, del_idx in enumerate(del_idx_list):
            #     del_idx -= counter
            #     del imgps_x_one_img[del_idx]
            #     del imgps_y_one_img[del_idx]

            img_color = cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)
            img_cnt = cv.drawContours(img_color, valid_contours, -1, (255, 0, 0), 5)
            plt.imshow(img_cnt)
            plt.show()

            df = pd.DataFrame({"imgps_x": imgps_x_one_img, "imgps_y": imgps_y_one_img})
            file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir))
            file_path = os.path.join(file_path, "datasets\\Corners\\")
            df.to_csv(file_path + img_name + ".csv",
                      index=False, sep=',')

    def refine_contour(self, contour, center, img):
        if self.Nc == 3:
            moment_kernel = np.array([[0.2424, 0.4319, 0.2424],
                                      [0.4319, 0.4444, 0.4319],
                                      [0.2424, 0.4319, 0.2424]])

        refined_cnt = np.zeros_like(contour)
        for cnt in contour:
            for i, cnt_point_coord in enumerate(cnt):
                cnt_point_u = cnt_point_coord[0][0]
                cnt_point_v = cnt_point_coord[0][1]
                theta = np.arctan2(center[0]-cnt_point_u, center[1]-cnt_point_v)

                M00 = np.sum(img[cnt_point_u-1:cnt_point_u+2, cnt_point_v-1:cnt_point_v+2] * moment_kernel)

                h1 = np.mean(img[cnt_point_u-5:cnt_point_u-2, cnt_point_v-5:cnt_point_v-2])
                h2 = np.mean(img[cnt_point_u+2:cnt_point_u+5, cnt_point_v+2:cnt_point_v+5])

                S2 = (M00 - np.pi * h1) / (h2 - h1)

                beta = 0.5114 * S2 + 0.7674

                l = np.cos(beta)

                refined_cnt_u = cnt_point_u + self.Nc * l * np.cos(theta) / 2
                refined_cnt_v = cnt_point_v + self.Nc * l * np.sin(theta) / 2
                refined_cnt[i][0], refined_cnt[i][1] = refined_cnt_u, refined_cnt_v

        return refined_cnt


if __name__ == '__main__':
    test_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir))
    test_path = os.path.join(test_path, "datasets\\Images")

    circle_center_extraction = ExtractCircleCenter(test_path, 3, 3072, 4096)
    circle_center_extraction.extract_circle_center()
