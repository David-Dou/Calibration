import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def rebuild(rebuild_file, camera_mtx, camera_ex_param, dlp_mtx, dlp_ex_param,
            vis_color_interval, vis_height_interval):
    rebuild_ret_list = []

    print('\nStart rebuilding...')

    data = pd.read_csv(rebuild_file)
    u_c = data['imgps_x'].values
    v_c = data[' imgps_y'].values
    # phase_0 = data[' ph0'].values
    u_dlp = data[' u'].values

    imgpoints = np.concatenate((u_c, v_c)).reshape(2, -1)

    imgpoints_homo = np.concatenate((imgpoints, np.full((1, imgpoints.shape[1]), 1)), axis=0,
                                    dtype=np.float32)

    camera_coordinate = np.matmul(np.linalg.inv(camera_mtx), imgpoints_homo)

    coefficient_mat = np.zeros(shape=(3, 3), dtype=np.float32)
    ret = np.zeros(shape=(3, ), dtype=np.float32)
    rebuild_rets = np.zeros(shape=(3, imgpoints.shape[1]))
    dlp_param = np.matmul(dlp_mtx, dlp_ex_param)

    for i in range(0, camera_coordinate.shape[1]):
        x_c, y_c = camera_coordinate[0][i], camera_coordinate[1][i]

        coefficient_mat[0][0] = camera_ex_param[0][0]
        coefficient_mat[0][1] = camera_ex_param[0][1]
        coefficient_mat[0][2] = camera_ex_param[0][2]
        coefficient_mat[1][0] = camera_ex_param[1][0]
        coefficient_mat[1][1] = camera_ex_param[1][1]
        coefficient_mat[1][2] = camera_ex_param[1][2]
        coefficient_mat[2][0] = dlp_param[0][0] - dlp_param[2][0] * u_dlp[i]
        coefficient_mat[2][1] = dlp_param[0][1] - dlp_param[2][1] * u_dlp[i]
        coefficient_mat[2][2] = dlp_param[0][2] - dlp_param[2][2] * u_dlp[i]

        ret[0] = x_c - camera_ex_param[0][3]
        ret[1] = y_c - camera_ex_param[1][3]

        ret[2] = dlp_param[2][3] * u_dlp[i] - dlp_param[0][3]

        rebuild_ret = np.dot(np.linalg.inv(coefficient_mat), ret)

        rebuild_rets[:, i] = rebuild_ret

    rebuild_ret_list.append(rebuild_rets)

    def plot_3d_heatmap(rebuild_ret_list, threshs, color_list):
        for pred in rebuild_ret_list:
            pred_colored = []

            for thresh in threshs:
                # why wrong? and?
                # pred_idx.append(thresh[0] <= pred[2] < thresh[1])

                low_thresh = pred[2] >= thresh[0]
                high_thresh = pred[2] < thresh[1]

                pred_colored.append(pred[:, low_thresh & high_thresh])

            ax = plt.axes(projection='3d')
            for i in range(len(color_list)):
                pred_c = pred_colored[i]
                points_x_pred, points_y_pred, points_z_pred = pred_c[0], pred_c[1], pred_c[2]
                ax.scatter3D(points_x_pred, points_y_pred, points_z_pred, c=color_list[i], marker='.')

            plt.show()

    def plot_2d_heatmap(rebuild_ret_list):
        for pred in rebuild_ret_list:
            label_z = np.rint(pred[2] / 0.001)
            fig, ax = plt.subplots()
            ax0 = ax.scatter(pred[0], pred[1], c=label_z, marker='o', cmap='jet')
            fig.colorbar(ax0, label='Z/μm')

            ax.set_xlabel('X/mm')
            ax.set_ylabel('Y/mm')

            plt.savefig("results/2D_heatmap.png")
            plt.show()

    plot_3d_heatmap(rebuild_ret_list,
                    threshs=vis_height_interval,
                    color_list=vis_color_interval)

    plot_2d_heatmap(rebuild_ret_list)
