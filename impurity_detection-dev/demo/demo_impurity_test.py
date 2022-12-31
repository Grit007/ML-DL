# -*- coding:utf-8 -*-
from mmdet.apis import init_detector, inference_detector, show_result,show_result_1, save_result_1
import os,time
import cv2,sys
import numpy as np

config_file = '/root/project/pytorch/highspeed.v3/configs/hrnet/cascade_rcnn_hrnetv2p_w32_v2.py'
checkpoint_file = '/root/project/pytorch/highspeed.v3/ckpt_dirs/cascade_rcnn_hrnetv2p_w32_impurity_v2.bak/epoch_40.pth'
# config_file = '/root/project/pytorch/highspeed.v3/configs/retinanet_r50_fpn_1x.py'
# checkpoint_file = '/root/project/pytorch/highspeed.v3/ckpt_dirs/retinanet_r50_fpn_impurity/epoch_25.pth'

model = init_detector(config_file, checkpoint_file)

images_path = '/data/deeplearning/wdq/impurity_db/data/221.jpg'
save_path = '/data/deeplearning/wdq/'


# images_path = '/root/project/pytorch/highspeed.v3/verify_data_en/'
# save_path = '/root/project/pytorch/highspeed.v3/verify_data_v2.res/'

if not os.path.exists(save_path):
    os.mkdir(save_path)


start_time = time.time()
# num_images = len(os.listdir(images_path))
# file_path = os.path.join(images_path, image)
# img = cv2.imread(images_path)
# img = cv2.resize(img, (2048, 1024))
# output = inference_detector(model, img)
# # show_result(img, result, class_names, score_thr=0.1, out_file=None)
# show_result(img, output, ["1"], out_file=os.path.join(save_path, "test.jpg"))

# if img is not None:
#     h, w, _ = img.shape
#     print(h, w)
#     crop_l = img[:, :4096, :]
#     crop_r = img[:, w - 4096:, :]
#
#     crop_m = None
#     if w>4096*2:
#         crop_m = img[:, w//2 - 2048:w//2 + 2048, :]
#
#     crop_l = cv2.resize(crop_l, (2048, 1024))
#     output_l = inference_detector(model, crop_l)
#     bboxes_l = show_result_1(output_l, detla = 0)
#
#     crop_r = cv2.resize(crop_r, (2048, 1024))
#     output_r = inference_detector(model, crop_r)
#     bboxes_r = show_result_1(output_r, detla = w - 4096)
#
#     bboxes = bboxes_l
#     if len(bboxes)==0 and len(bboxes_r)>0:
#         bboxes = bboxes_r
#     elif len(bboxes)>0 and len(bboxes_r)>0:
#         bboxes = np.concatenate([bboxes, bboxes_r])
#
#     if crop_m is not None:
#         crop_m = cv2.resize(crop_m, (2048, 1024))
#         output_m = inference_detector(model, crop_m)
#         bboxes_m = show_result_1(output_m, detla = w/2 - 2048)
#
#         if len(bboxes) == 0 and len(bboxes_m) > 0:
#             bboxes = bboxes_m
#         elif len(bboxes) > 0 and len(bboxes_m) > 0:
#             bboxes = np.concatenate([bboxes, bboxes_r])
#
#     save_result_1(img, bboxes, ['1'], score_thr=0.3, out_file=os.path.join(save_path, image))


# print("Time: {:.2f} s / img".format((time.time() - start_time)/num_images))
