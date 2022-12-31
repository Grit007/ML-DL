# -*- coding:utf-8 -*-
from mmdet.apis import init_detector, inference_detector, write_result
import os,time
import cv2,json
import numpy as np

config_file = '/root/project/pytorch/highspeed.v3/configs/hrnet/cascade_rcnn_hrnetv2p_w32_v2.py'
checkpoint_file = '/root/project/pytorch/highspeed.v3/ckpt_dirs/cascade_rcnn_hrnetv2p_w32_impurity_v2.bak/epoch_40.pth'
# config_file = '/root/project/pytorch/highspeed.v3/configs/retinanet_r50_fpn_1x.py'
# checkpoint_file = '/root/project/pytorch/highspeed.v3/ckpt_dirs/retinanet_r50_fpn_impurity/epoch_25.pth'

model = init_detector(config_file, checkpoint_file)

images_path = '/data/deeplearning/wdq/impurity_db/data/impurity_test'
save_path = '/data/deeplearning/wdq/impurity.data.res.test/'


# images_path = '/root/project/pytorch/highspeed.v3/verify_data_en/'
# save_path = '/root/project/pytorch/highspeed.v3/verify_data_v2.res/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

start_time = time.time()
num_images = len(os.listdir(images_path))

det_results = {}
results = []
src_path = [
    {
        "id": "334",
        "path": "/data/deeplearning/wdq/impurity_db/data/impurity_test/334.jpg",
        "base64": ""
    },
    {
        "id": "469",
        "path": "/data/deeplearning/wdq/impurity_db/data/impurity_test/469.jpg",
        "base64": ""
    },
    {
        "id": "10",
        "path": "/data/deeplearning/wdq/impurity_db/data/impurity_test/10.jpg",
        "base64": ""
    },
    {
        "id": "54",
        "path": "/data/deeplearning/wdq/impurity_db/data/impurity_test/54.jpg",
        "base64": ""
    }
]
try:
    for image in src_path:
        print(image)
        image_path = image['path']
        if os.path.exists(image_path):
            # file_path = os.path.join(images_path, image)
            img = cv2.imread(image_path)
        else:
            continue
        if img is not None:
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
            if w>4096:
                crop_r = img[:, w - 4096:, :]

            crop_m = None
            if w>4096*2:
                crop_m = img[:, w//2 - 2048:w//2 + 2048, :]

            if crop_l is not None:
                crop_l = cv2.resize(crop_l, (2048, 1024))
                output_l = inference_detector(model, crop_l)
                bboxes_l = write_result(output_l, detla = 0, l_use_scale=l_use_scale, src_size=(h, w))
            else:
                bboxes_l = []

            if crop_r is not None:
                crop_r = cv2.resize(crop_r, (2048, 1024))
                output_r = inference_detector(model, crop_r)
                bboxes_r = write_result(output_r, detla = w - 4096)
            else:
                bboxes_r = []

            if crop_m is not None:
                crop_m = cv2.resize(crop_m, (2048, 1024))
                output_m = inference_detector(model, crop_m)
                bboxes_m = write_result(output_m, detla = w/2 - 2048)
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

            per_image_res['id'] = image['id']
            per_image_res_box = []

            if isinstance(bboxes, np.ndarray):
                bboxes_list = bboxes.tolist()
                for per_box in bboxes_list:
                    per_box_res = {}
                    per_box_res['box'] = {"xmin": int(per_box[0]), "ymin": int(per_box[1]),
                                          "xmax": int(per_box[2]), "ymax": int(per_box[3])}
                    per_box_res['type'] = 1
                    per_box_res['score'] = per_box[-1]
                    per_image_res_box.append(per_box_res)

            per_image_res['bbox'] = per_image_res_box

            results.append(per_image_res)
    det_results['results']=results
    det_results['msg']="success"
    det_results['status']=1200
except:
    det_results['msg']="failed"
    det_results['status']=1400
    
print("Time: {:.2f} s / img".format((time.time() - start_time)/num_images))

res_jsonfile = '/data/deeplearning/wdq/det_results.json'
with open(res_jsonfile, 'w') as json_file:
    json_file.write(json.dumps(det_results, ensure_ascii=False))
    json_file.close()

    