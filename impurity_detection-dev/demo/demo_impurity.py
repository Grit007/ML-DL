# -*- coding:utf-8 -*-
from mmdet.apis import init_detector, inference_detector, write_result
import os, time
import cv2, json
import numpy as np

config_file = 'D:\\code\\highspeed.v3\\configs\\hrnet\\cascade_rcnn_hrnetv2p_w32_impurity_v5'
checkpoint_file = 'D:\\highspeed.v3\\weights\\epoch_40.pth'


model = init_detector(config_file, checkpoint_file)

images_path = 'E:\\down\\20191021'
save_path = 'E:\\data_res\\20191021'


if not os.path.exists(save_path):
    os.makedirs(save_path)

counter = 0

image_list = []
unit_list = []
for image in os.listdir(images_path):
    counter += 1
    per_image_path = os.path.join(images_path, image)
    unit_list.append(per_image_path)
    if counter%4==0:
        image_list.append(unit_list)
        unit_list = []

counter = 0
for idx, image_unit in enumerate(image_list):
    unit_save_path = os.path.join(save_path, str(idx + 1))
    if unit_save_path is not os.path.exists(unit_save_path):
        os.makedirs(unit_save_path)

    package_results = {}
    det_results = {}

    for per_image_path in image_unit:
        if os.path.exists(per_image_path):
            img = cv2.imread(per_image_path)
        else:
            continue
        if img is not None:
            image_name = os.path.split(per_image_path)[1]
            cv2.imwrite(os.path.join(unit_save_path, image_name), img)
            per_image_res = {}
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
            if w > 4096:
                crop_r = img[:, w - 4096:, :]

            crop_m = None
            if w > 4096 * 2:
                crop_m = img[:, w // 2 - 2048:w // 2 + 2048, :]

            if crop_l is not None:
                crop_l = cv2.resize(crop_l, (2048, 1024))
                output_l = inference_detector(model, crop_l)
                bboxes_l = write_result(output_l, detla=0, l_use_scale=l_use_scale, src_size=(h, w))
            else:
                bboxes_l = []

            if crop_r is not None:
                crop_r = cv2.resize(crop_r, (2048, 1024))
                output_r = inference_detector(model, crop_r)
                bboxes_r = write_result(output_r, detla=w - 4096)
            else:
                bboxes_r = []

            if crop_m is not None:
                crop_m = cv2.resize(crop_m, (2048, 1024))
                output_m = inference_detector(model, crop_m)
                bboxes_m = write_result(output_m, detla=w / 2 - 2048)
            else:
                bboxes_m = []

            bboxes = []
            if isinstance(bboxes_l, np.ndarray):
                bboxes = bboxes_l

            if isinstance(bboxes_r, np.ndarray):
                if isinstance(bboxes, np.ndarray):
                    bboxes = np.concatenate([bboxes, bboxes_r])
                else:
                    bboxes = bboxes_r

            if isinstance(bboxes_m, np.ndarray):
                if isinstance(bboxes, np.ndarray):
                    bboxes = np.concatenate([bboxes, bboxes_m])
                else:
                    bboxes = bboxes_m

            per_image_res_box = []
            if isinstance(bboxes, np.ndarray):
                bboxes_list = bboxes.tolist()
                for box_idx, per_box in enumerate(bboxes_list):
                    per_box_res = {}
                    per_box_res['bbox'] = {"xmin": int(per_box[0]), "ymin": int(per_box[1]),
                                           "xmax": int(per_box[2]), "ymax": int(per_box[3])}

                    per_box_res['polygon'] = [int(per_box[0]), int(per_box[1]),
                                              int(per_box[2]), int(per_box[1]),
                                              int(per_box[2]), int(per_box[3]),
                                              int(per_box[0]), int(per_box[3]),
                                              int(per_box[0]), int(per_box[1])
                                              ]

                    per_box_res['id'] = box_idx
                    per_box_res['category'] = "2"
                    per_image_res_box.append(per_box_res)

                per_image_res['objects'] = per_image_res_box
                per_image_res['path'] = image_name
                det_results[image_name] = per_image_res
    package_results['imgs'] = det_results
    package_results['types'] = []

    res_jsonfile = os.path.join(unit_save_path, str(idx + 1) + '.json')
    with open(res_jsonfile, 'w') as json_file:
        json_file.write(json.dumps(package_results, ensure_ascii=False))
        json_file.close()

            
# for image in os.listdir(images_path):
#     print(image)
#     per_image_path = os.path.join(images_path, image)
#     if os.path.exists(per_image_path):
#         img = cv2.imread(per_image_path)
#     else:
#         continue
#     if img is not None:
#         per_image_res = {}
#         h, w, _ = img.shape
#         print(h, w)
#
#         crop_l = None
#         l_use_scale = False
#         if w <= 4096:
#             crop_l = img
#             l_use_scale = True
#         else:
#             crop_l = img[:, :4096, :]
#
#         crop_r = None
#         if w > 4096:
#             crop_r = img[:, w - 4096:, :]
#
#         crop_m = None
#         if w > 4096 * 2:
#             crop_m = img[:, w // 2 - 2048:w // 2 + 2048, :]
#
#         if crop_l is not None:
#             crop_l = cv2.resize(crop_l, (2048, 1024))
#             output_l = inference_detector(model, crop_l)
#             bboxes_l = write_result(output_l, detla=0, l_use_scale=l_use_scale, src_size=(h, w))
#         else:
#             bboxes_l = []
#
#         if crop_r is not None:
#             crop_r = cv2.resize(crop_r, (2048, 1024))
#             output_r = inference_detector(model, crop_r)
#             bboxes_r = write_result(output_r, detla=w - 4096)
#         else:
#             bboxes_r = []
#
#         if crop_m is not None:
#             crop_m = cv2.resize(crop_m, (2048, 1024))
#             output_m = inference_detector(model, crop_m)
#             bboxes_m = write_result(output_m, detla=w / 2 - 2048)
#         else:
#             bboxes_m = []
#
#         bboxes = []
#         if isinstance(bboxes_l, np.ndarray):
#             bboxes = bboxes_l
#
#         if isinstance(bboxes_r, np.ndarray):
#             if isinstance(bboxes, np.ndarray):
#                 bboxes = np.concatenate([bboxes, bboxes_r])
#             else:
#                 bboxes = bboxes_r
#
#         if isinstance(bboxes_m, np.ndarray):
#             if isinstance(bboxes, np.ndarray):
#                 bboxes = np.concatenate([bboxes, bboxes_m])
#             else:
#                 bboxes = bboxes_m
#
#         per_image_res_box = []
#         if isinstance(bboxes, np.ndarray):
#             bboxes_list = bboxes.tolist()
#             for idx, per_box in enumerate(bboxes_list):
#                 per_box_res = {}
#                 per_box_res['bbox'] = {"xmin": int(per_box[0]), "ymin": int(per_box[1]),
#                                        "xmax": int(per_box[2]), "ymax": int(per_box[3])}
#
#                 per_box_res['polygon'] = [int(per_box[0]), int(per_box[1]),
#                                           int(per_box[2]), int(per_box[1]),
#                                           int(per_box[2]), int(per_box[3]),
#                                           int(per_box[0]), int(per_box[3]),
#                                           int(per_box[0]), int(per_box[1])
#                                           ]
#
#                 per_box_res['id'] = idx
#                 per_box_res['category'] = "4"
#                 per_image_res_box.append(per_box_res)
#
#             per_image_res['objects'] = per_image_res_box
#             per_image_res['path'] = image
#
#         pur_unit_path = os.path.join(save_path, str(counter%4 + 1))
#         if not os.path.exists(pur_unit_path):
#             os.makedirs(pur_unit_path)
#         cv2.imwrite(os.path.join(pur_unit_path, image), img)
#         counter += 1
#
#         det_results[image] = per_image_res
#         cv2.imwrite(os.path.join(pur_unit_path, image), img)
#     if counter%4 == 0:
#         package_results['imgs'] = det_results
#         package_results['types'] = []
#
#         pur_unit_path = os.path.join(save_path, str(counter))
#
#         res_jsonfile = os.path.join(pur_unit_path, str(counter)+'.json')
#         with open(res_jsonfile, 'w') as json_file:
#             json_file.write(json.dumps(package_results, ensure_ascii=False))
#             json_file.close()
#         package_results = {}
#         det_results = {}

