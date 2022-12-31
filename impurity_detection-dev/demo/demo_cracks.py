from mmdet.apis import init_detector, inference_detector, show_result, json_result
import os,time
import cv2,json

config_file = '/root/project/pytorch/highspeed.v3/configs/hrnet/cascade_rcnn_hrnetv2p_w32_cracks.py'
checkpoint_file = '/root/project/pytorch/highspeed.v3/ckpt_dirs/cascade_rcnn_hrnetv2p_w32_cracks/epoch_20.pth'

model = init_detector(config_file, checkpoint_file)

images_path = '/data/deeplearning/wdq/orthographic_datasets/204359490_20191006110809380/orthographic_coarse'

track_id = os.path.split(images_path)[-1]
trackid_res = {}
start_time = time.time()
for image in os.listdir(images_path):
    if image.endswith('.jpg') and image.startswith('projection'):
        img = cv2.imread(os.path.join(images_path, image))
        per_image_attr = {}
        image_info = {}
        per_image_attr["trackId"] = track_id
        per_image_attr["trackPointId"] = image[11:-4]
        per_image_result=[]

        if img is not None:
            ori_w = img.shape[1]
            ori_h = img.shape[0]
            resize_img = cv2.resize(img, (1024, 960))
            resize_w = resize_img.shape[1]
            resize_h = resize_img.shape[0]
            output = inference_detector(model, img)
            per_image_result, num_boxes = json_result(output, resize_scale=(ori_w/resize_w, ori_h/resize_h))
        else:
            pass
        per_image_attr["objects"]=per_image_result

        image_info["base"] = {"image_seq": "004"}

        if num_boxes>0:
            image_info["has_cracks"] = True
        else:
            image_info["has_cracks"] = False

        per_image_attr["image_info"] = image_info

        trackid_res[image] = per_image_attr

outputs = {}
outputs["imgs"]= trackid_res
jsonfile = '/data/deeplearning/wdq/orthographic_datasets/204359490_20191006110809380/crack_detection.json'

with open(jsonfile, 'w') as json_file:
    # print(outputs)
    json_file.write(json.dumps(outputs, ensure_ascii=False))