# -*- coding:utf-8 -*-
from mmdet.apis import init_detector, inference_detector, show_result,show_result_1, save_result_1
import os,time
import cv2,sys
import numpy as np

config_file = 'D:\\highspeed.v3\\weights\\faster_rcnn_1127\\faster_rcnn_hrnetv2p_w32_1x.py'
checkpoint_file = 'D:\\highspeed.v3\\weights\\faster_rcnn_1127\\epoch_16.pth'
# config_file = '/root/project/pytorch/highspeed.v3/configs/retinanet_r50_fpn_1x.py'
# checkpoint_file = '/root/project/pytorch/highspeed.v3/ckpt_dirs/retinanet_r50_fpn_impurity/epoch_25.pth'

model = init_detector(config_file, checkpoint_file)

images_path = 'E:\\down\\20191021'
save_path = 'E:\\20191021\\data_faster_rnn_w32_16'


# images_path = '/root/project/pytorch/highspeed.v3/verify_data_en/'
# save_path = '/root/project/pytorch/highspeed.v3/verify_data_v2.res/'

if not os.path.exists(save_path):
    os.makedirs(save_path)


start_time = time.time()
num_images = len(os.listdir(images_path))

num_imgs_processed = 0
num_imgs_ouput = 0

for folder in os.walk(images_path):
    for file in folder[2]:
        file_path = folder[0]+'\\' + file
        # here only for goods-train impurity detection
        if file_path.endswith('P.jpg') or file_path.endswith('U.jpg'):
            num_imgs_processed += 1
            image = os.path.split(file_path)[1]
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

                # only output the image with impurity
                if len(bboxes) > 0:
                    save_result_1(img, bboxes, ['1'], score_thr=0.3, out_file=os.path.join(save_path, image))
                    num_imgs_ouput += 1

print("Time: {:.2f} s / img".format((time.time() - start_time)/num_images))
print("Process Img:%d,  Output Img:%d" % (num_imgs_processed, num_imgs_ouput))
