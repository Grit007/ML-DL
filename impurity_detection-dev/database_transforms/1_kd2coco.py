import os
import json
import numpy as np
import cv2

def kd2coco(path):
    info = {'description': 'kd training data converted to coco format.',
            'url': '',
            'version': '1.0',
            'year': 2019,
            'contributor': 'kd',
            'date_created': '2019-07-05 09:11:52.357475'}

    licenses = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                 'id': 1,
                 'name': 'Attribution-NonCommercial-ShareAlike License'}]

    categories = [{
        "supercategory": "node",
        "id": 1,
        "name": "node"}
    ]

    test_annotations = []
    train_annotations = []
    train_images = []
    test_images = []

    train_f_anchors = open(
        "/data/deeplearning/wdq/high_speed_single_cls.v2/json/train.txt", 'w')
    test_f_anchors = open(
        "/data/deeplearning/wdq/high_speed_single_cls.v2/json/val.txt", 'w')
    num_images = 0
    num_img = 0
    image_id = 10000
    counter = 0
    target_v1_counter = 0
    for folder in os.walk(path):
        for file in folder[2]:
            file_path = folder[0]+'/'+file
            if file_path.endswith('.json'):
                with open(file_path) as f:
                    kd = json.load(f)

                for img in kd['imgs']:
                    num_img += 1
                    random = np.random.uniform(0, 1)
                    
                    src = cv2.imread(os.path.join(folder[0], img), flags=0)
                    size = src.shape
                    print(os.path.join(folder[0], img))

                    image_id += 1
                    image_1_id = image_id
                    image_1_name = img[:-4] + str(image_id) + '.jpg'
                    image_1 = {'license': 1,
                               'file_name': str(0) * 7 + str(image_id) + '.jpg',
                               'coco_url': '',
                               'height': 1100,
                               'width': 1630,
                               'date_captured': '2019-07-05 09:11:52.357475',
                               'flickr_url': '',
                               'id': image_id}

                    image_id += 1
                    image_2_id = image_id
                    image_2 = {'license': 1,
                               'file_name': str(0) * 7 + str(image_id) + '.jpg',
                               'coco_url': '',
                               'height': 1100,
                               'width': 1630,
                               'date_captured': '2019-07-05 09:11:52.357475',
                               'flickr_url': '',
                               'id': image_id}

                    num_boxes_per_image = 0

                    for obj in kd['imgs'][img]['objects']:
                        if int(obj['category'])>0:
                            class_id = 1
                        else:
                            class_id = 0
                            
                        if class_id > 0:
                            target_v1_counter += 1
                            if size[0] == 4384:
                                xmin = max(
                                    float(float(int(obj['bbox']['xmin']))*(1630./6576)), 0)
                                ymin = max(
                                    float(float(int(obj['bbox']['ymin']))*(1100./4384)), 0)
                                xmax = min(
                                    float(float(int(obj['bbox']['xmax']))*(1630./6576)), 1630)
                                ymax = min(
                                    float(float(int(obj['bbox']['ymax']))*(1100./4384)), 1100)
                            if size[0] == 3248:
                                xmin = max(
                                    float(float(int(obj['bbox']['xmin']))*(1630./4872)), 0)
                                ymin = max(
                                    float(float(int(obj['bbox']['ymin']))*(1100./3248)), 0)
                                xmax = min(
                                    float(float(int(obj['bbox']['xmax']))*(1630./4872)), 1630)
                                ymax = min(
                                    float(float(int(obj['bbox']['ymax']))*(1100./3248)), 1100)
                            if size[0] == 1230:
                                xmin = max(
                                    float(float(int(obj['bbox']['xmin']))*(1630./1620)), 0)
                                ymin = max(
                                    float(float(int(obj['bbox']['ymin']))*(1100./1230)), 0)
                                xmax = min(
                                    float(float(int(obj['bbox']['xmax']))*(1630./1620)), 1630)
                                ymax = min(
                                    float(float(int(obj['bbox']['ymax']))*(1100./1230)), 1100)
                            print(size, xmax-xmin, ymax-ymin, 1.0*(xmax-xmin)/(ymax-ymin))

                            m_xmin = 1630 - xmax
                            m_ymin = ymin
                            m_xmax = 1630 - xmin
                            m_ymax = ymax
                            len_h = ymax - ymin
                            len_w = xmax - xmin
                            polygon = [xmin, ymin,
                                           xmax, ymin,
                                           xmax, ymax,
                                           xmin, ymax]
                            m_polygon = [m_xmin, m_ymin,
                                             m_xmax, m_ymin,
                                             m_xmax, m_ymax,
                                             m_xmin, m_ymax]
                            valid_obj = False
                            if class_id == 1:
                                if len_h >= 15 and len_w >= 15:
                                    valid_obj = True
                                    target_v1_counter += 1
                                if valid_obj:
                                    num_boxes_per_image += 1
                                    counter += 1
                                    annotation = {'segmentation': [polygon],
                                                  'area': len_w*len_h,
                                                  'iscrowd': 0,
                                                  'image_id': image_1_id,
                                                  'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
                                                  'category_id': int(class_id),
                                                  'id': counter}
                                    counter += 1
                                    m_annotation = {'segmentation': [m_polygon],
                                                    'area': len_w*len_h,
                                                    'iscrowd': 0,
                                                    'image_id': image_2_id,
                                                    'bbox': [m_xmin, m_ymin, m_xmax-m_xmin, m_ymax-m_ymin],
                                                    'category_id': int(class_id),
                                                    'id': counter}
                                    if random <= 0.9:
                                        train_annotations.append(annotation)
                                        train_annotations.append(m_annotation)
                                    else:
                                        test_annotations.append(annotation)
                                        test_annotations.append(m_annotation)

                    if num_boxes_per_image >= 1:
                        num_images += 1
                        if random <= 0.9:
                            train_images.append(image_1)
                            train_images.append(image_2)
                            train_f_anchors.write(folder[0] + '/' + image_1_name + '\n')
                        else:
                            test_images.append(image_1)
                            test_images.append(image_2)
                            test_f_anchors.write(folder[0] + '/' + image_1_name + '\n')
                    else:
                        pass
    print('num_images:', num_images, num_img)
    print('class_num:', target_v1_counter)
    return {'info': info, 'images': train_images, 'licenses': licenses, 'annotations': train_annotations, 'categories': categories}, \
        {'info': info, 'images': test_images, 'licenses': licenses,
         'annotations': test_annotations, 'categories': categories}

                          
train_label_dict, test_label_dict = kd2coco(
    '/data/deeplearning/wdq/high_speed_db/20190828_2007')

train_jsonfile = '/data/deeplearning/wdq/high_speed_single_cls.v2/annotations/train.json'
with open(train_jsonfile, 'w') as json_file:
    json_file.write(json.dumps(train_label_dict, ensure_ascii=False))
    json_file.close()

test_jsonfile = '/data/deeplearning/wdq/high_speed_single_cls.v2/annotations/val.json'
with open(test_jsonfile, 'w') as json_file:
    json_file.write(json.dumps(test_label_dict, ensure_ascii=False))
    json_file.close()
