


import cv2
import os
import numpy as np


def compute(path):
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    counter = 0
    for file_name in file_names:
        counter += 1
        img = cv2.imread(os.path.join(path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        R_channel = R_channel + np.sum(np.power(img[:, :, 0]-R_mean, 2))
        G_channel = G_channel + np.sum(np.power(img[:, :, 1]-G_mean, 2))
        B_channel = B_channel + np.sum(np.power(img[:, :, 2]-B_mean, 2))
    w, h = img.shape[:2]

    R_std = np.sqrt(R_channel/(counter*w*h))
    G_std = np.sqrt(G_channel/(counter*w*h))
    B_std = np.sqrt(B_channel/(counter*w*h))

    print("R_mean/G_mean/B_mean:", R_mean, G_mean, B_mean)
    print("R_std/G_std/B_std:", R_std, G_std, B_std)


if __name__ == '__main__':
    compute(
        path='/data/deeplearning/wdq/high_speed_single_cls.v2/train/')
