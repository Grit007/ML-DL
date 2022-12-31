import os
import json
import numpy as np
import cv2,copy


def kd2coco(src_path, train_save_path, test_save_path):
    info = {'description': 'kd training data converted to coco format.',
            'url': '',
            'version': '1.0',
            'year': 2019,
            'contributor': 'kd',
            'date_created': '2019-10-12 09:11:52.357475'}

    licenses = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                 'id': 1,
                 'name': 'Attribution-NonCommercial-ShareAlike License'}]

    categories = [
        {
            "supercategory": "pavement_cracks",
            "id": 1,
            "name": "pavement_cracks"}
    ]

    train_annotations = []
    test_annotations = []
    train_images = []
    test_images = []
    counter =0
    image_id = 10000
    for folder in os.walk(src_path):
        for file in folder[2]:
            file_path = folder[0] + '/' + file
            if file_path.endswith('.json'):
                with open(file_path) as f:
                    kd = json.load(f)
                for img in kd['imgs']:
                    valid = False
                    print(img)
                    ori_img = cv2.imread(os.path.join(folder[0], img))
                    h, w, _ = ori_img.shape
                    print(h, w)
                    if ori_img is None:
                        continue
                    else:
                        src_img = cv2.resize(ori_img, dsize=(1024, 960))
                        random = np.random.uniform(0, 1)
                        n_h, n_w, _=src_img.shape
                        # print( n_h, n_w)

                        image_id += 1
                        image_1_id = image_id
                        image_info_r = {'license': 1,
                                   'file_name': str(0) * 7 + str(image_1_id) + '.jpg',
                                   'coco_url': '',
                                   'height': 960,
                                   'width': 1024,
                                   'date_captured': '2019-08-20 09:11:52.357475',
                                   'flickr_url': '',
                                   'id': image_id}

                        image_id += 1
                        image_2_id = image_id
                        image_info_s = {'license': 1,
                                   'file_name': str(0) * 7 + str(image_2_id) + '.jpg',
                                   'coco_url': '',
                                   'height': 960,
                                   'width': 1024,
                                   'date_captured': '2019-08-20 09:11:52.357475',
                                   'flickr_url': '',
                                   'id': image_id}

                        if len(kd['imgs'][img]['objects']) > 0:
                            for obj in kd['imgs'][img]['objects']:
                                if 'bbox' in obj and obj['category'] == "1":
                                    xmin = max(int(obj['bbox']['xmin'])*1024.0/w, 0)
                                    ymin = max(int(obj['bbox']['ymin'])*960.0/h, 0)
                                    xmax = min(int(obj['bbox']['xmax'])*1024.0/w, 1024)
                                    ymax = min(int(obj['bbox']['ymax'])*960.0/h, 960)

                                    m_xmin = 1024 - xmax
                                    m_ymin = ymin
                                    m_xmax = 1024 - xmin
                                    m_ymax = ymax

                                    len_w = xmax - xmin
                                    len_h = ymax - ymin
                                    if len_h>10 and len_w>10:
                                        valid = True
                                        counter += 1
                                        annotation_r = {'segmentation': [[xmin, ymin],[xmax, ymin],
                                                                       [xmax, ymax],[xmin, ymax]],
                                                      'area': len_w*len_h,
                                                      'iscrowd': 0,
                                                      'image_id': image_1_id,
                                                      'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                                                      'category_id': int(1),
                                                      'id': counter}

                                        counter += 1
                                        annotation_s = {'segmentation': [[m_xmin, m_ymin],[m_ymax, m_ymin],
                                                                         [m_xmax, m_ymax],[m_xmin, m_ymax]],
                                                      'area': len_w*len_h,
                                                      'iscrowd': 0,
                                                      'image_id': image_2_id,
                                                      'bbox': [m_xmin, m_ymin, m_xmax - m_xmin, m_ymax - m_ymin],
                                                      'category_id': int(1),
                                                      'id': counter}

                                        if random <= 0.95:
                                            train_annotations.append(annotation_r)
                                            train_annotations.append(annotation_s)
                                        else:
                                            test_annotations.append(annotation_r)
                                            test_annotations.append(annotation_s)
                    if valid:
                        mirror_img = copy.deepcopy(src_img)
                        for wi in range(n_w):
                            mirror_img[:, n_w - wi - 1] = src_img[:, wi]

                        if random <= 0.95:
                            train_images.append(image_info_r)
                            train_images.append(image_info_s)
                            cv2.imwrite(os.path.join(train_save_path, str(0) * 7 + str(image_1_id) + '.jpg'), src_img)
                            cv2.imwrite(os.path.join(train_save_path, str(0) * 7 + str(image_2_id) + '.jpg'), mirror_img)
                        else:
                            test_images.append(image_info_r)
                            test_images.append(image_info_s)
                            cv2.imwrite(os.path.join(test_save_path, str(0) * 7 + str(image_1_id) + '.jpg'), src_img)
                            cv2.imwrite(os.path.join(test_save_path, str(0) * 7 + str(image_2_id) + '.jpg'), mirror_img)


    print(image_id, counter)
    return {'info': info, 'images': train_images, 'licenses': licenses, 'annotations': train_annotations, 'categories': categories}, \
            {'info': info, 'images': test_images, 'licenses': licenses, 'annotations': test_annotations, 'categories': categories},

train_label_dict, test_label_dict=kd2coco(src_path="/data/deeplearning/wdq/orthographic_datasets/orthographic_cracks_db.v2",
        train_save_path="/data/deeplearning/wdq/road_disease_single_cls_cracks_orthographic.v2/single_cls_train_coco",
        test_save_path="/data/deeplearning/wdq/road_disease_single_cls_cracks_orthographic.v2/single_cls_val_coco")

train_jsonfile = '/data/deeplearning/wdq/road_disease_single_cls_cracks_orthographic.v2/src_annotations/train.json'
with open(train_jsonfile, 'w') as json_file:
    json_file.write(json.dumps(train_label_dict, ensure_ascii=False))
    json_file.close()

val_jsonfile = '/data/deeplearning/wdq/road_disease_single_cls_cracks_orthographic.v2/src_annotations/val.json'
with open(val_jsonfile, 'w') as json_file:
    json_file.write(json.dumps(test_label_dict, ensure_ascii=False))
    json_file.close()