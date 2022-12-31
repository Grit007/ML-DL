from mmdet.apis import init_detector, inference_detector, show_result
import os,time
import cv2


config_file = '/root/project/pytorch/highspeed.v3/configs/hrnet/cascade_rcnn_hrnetv2p_w32.py'
checkpoint_file = '/root/project/pytorch/highspeed.v3/ckpt_dirs/cascade_rcnn_hrnetv2p_w32/epoch_24.pth'

model = init_detector(config_file, checkpoint_file)

# images_path = '/data/deeplearning/wdq/highspeed.test'
# save_path = '/data/deeplearning/wdq/highspeed.res.v4/'

images_path = '/data/deeplearning/wdq/high_speed_single_cls.v2/'
save_path = '/data/deeplearning/wdq/high_speed_single_cls.res.v3/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

start_time = time.time()
for folder in os.walk(images_path):
    for file in folder[2]:
        file_path = folder[0] + '/' + file
        if file_path.endswith('.jpg'):
            print(file_path)
            img = cv2.imread(file_path)
            if img is not None:
                w = img.shape[1]
                h = img.shape[0]
                img = cv2.resize(img, (1630, 1100))
                output = inference_detector(model, img)
                show_result(img, output, model.CLASSES, out_file=os.path.join(save_path, file), score_thr=0.250)

