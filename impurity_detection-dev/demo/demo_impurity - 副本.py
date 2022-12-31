# -*- coding:utf-8 -*-
from mmdet.apis import init_detector, inference_detector, show_result,show_result_1, save_result_1
import os,time
import cv2,sys
import numpy as np

config_file = 'D:\highspeed.v3\configs\hrnet\cascade_rcnn_hrnetv2p_w32_v2.py'
checkpoint_file = 'D:\highspeed.v3\weights\epoch_40.pth'
# config_file = '/root/project/pytorch/highspeed.v3/configs/retinanet_r50_fpn_1x.py'
# checkpoint_file = '/root/project/pytorch/highspeed.v3/ckpt_dirs/retinanet_r50_fpn_impurity/epoch_25.pth'

model = init_detector(config_file, checkpoint_file)

images_path = 'E:\\down\\20191021'
save_path = 'E:\\data_res\\20191021'


# images_path = '/root/project/pytorch/highspeed.v3/verify_data_en/'
# save_path = '/root/project/pytorch/highspeed.v3/verify_data_v2.res/'

if not os.path.exists(save_path):
    os.makedirs(save_path)


start_time = time.time()
num_images = len(os.listdir(images_path))

for image in os.listdir(images_path):
    print(image)
    file_path = os.path.join(images_path, image)
    if 1:
        img = cv2.imread(file_path)
        if img is not None:
            h, w, _ = img.shape
            print(h, w)

            crop_l = None
            l_use_scale = False
            if w <= 4096:
                crop_l = img
                l_use_scale = True
            else:
                crop_l = img[:, :4096, :]

            crop_r = None
            if w>4096:
                crop_r = img[:, w - 4096:, :]

            crop_m = None
            if w>4096*2:
                crop_m = img[:, w//2 - 2048:w//2 + 2048, :]

            crop_l = cv2.resize(crop_l, (2048, 1024))
            output_l = inference_detector(model, crop_l)
            bboxes_l = show_result_1(output_l, detla = 0, l_use_scale=l_use_scale, src_size=(h, w))

            if crop_r is not None:
                crop_r = cv2.resize(crop_r, (2048, 1024))
                output_r = inference_detector(model, crop_r)
                bboxes_r = show_result_1(output_r, detla = w - 4096)
            else:
                bboxes_r = []

            bboxes = bboxes_l
            if len(bboxes)==0 and len(bboxes_r)>0:
                bboxes = bboxes_r
            elif len(bboxes)>0 and len(bboxes_r)>0:
                bboxes = np.concatenate([bboxes, bboxes_r])

            if crop_m is not None:
                crop_m = cv2.resize(crop_m, (2048, 1024))
                output_m = inference_detector(model, crop_m)
                bboxes_m = show_result_1(output_m, detla = w/2 - 2048)

                if len(bboxes) == 0 and len(bboxes_m) > 0:
                    bboxes = bboxes_m
                elif len(bboxes) > 0 and len(bboxes_m) > 0:
                    bboxes = np.concatenate([bboxes, bboxes_r])

            save_result_1(img, bboxes, ['1'], score_thr=0.3, out_file=os.path.join(save_path, image))


print("Time: {:.2f} s / img".format((time.time() - start_time)/num_images))
