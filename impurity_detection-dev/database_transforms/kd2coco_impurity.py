import cv2
import os
import json

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
        "supercategory": "impurity",
        "id": 1,
        "name": "impurity"},
    {
        "supercategory": "bg",
        "id": 2,
        "name": "bg"},
    {
        "supercategory": "other",
        "id": 3,
        "name": "other"}
]

train_annotations = []
test_annotations = []
train_images = []
test_images = []
src_path = "G:\\impurity\\label_data\\label_class_20191204"

save_path = "G:\\impurity\\train_data\\20191204"
if not os.path.exists(save_path):
    os.makedirs(save_path)

counter = 0
image_id = 10000
for folder in os.walk(src_path):
    for file in folder[2]:
        file_path = folder[0] + '/' + file
        if file.endswith('.json'):
            with open(file_path) as f:
                kd = json.load(f)
            for img in kd['imgs']:
                ori_img = cv2.imread(os.path.join(folder[0], img))
                if ori_img is not None:
                    h, w, _ = ori_img.shape
                    crop_l = ori_img[:, :4096, :]
                    crop_r = ori_img[:, w-4096:, :]
                    crop_m = ori_img[:, w//2-2048: w//2+2048, :]

                    valid_l = False
                    valid_r = False
                    valid_m = False

                    image_id += 1
                    image_l_id = image_id
                    image_info_l = {'license': 1,
                                    'file_name': str(0) * 7 + str(image_l_id) + '.jpg',
                                    'coco_url': '',
                                    'height': 1024,
                                    'width': 2048,
                                    'date_captured': '2019-08-20 09:11:52.357475',
                                    'flickr_url': '',
                                    'id': image_id}

                    image_id += 1
                    image_r_id = image_id
                    image_info_r = {'license': 1,
                                    'file_name': str(0) * 7 + str(image_r_id) + '.jpg',
                                    'coco_url': '',
                                    'height': 1024,
                                    'width': 2048,
                                    'date_captured': '2019-08-20 09:11:52.357475',
                                    'flickr_url': '',
                                    'id': image_id}

                    image_id += 1
                    image_m_id = image_id
                    image_info_m = {'license': 1,
                                    'file_name': str(0) * 7 + str(image_m_id) + '.jpg',
                                    'coco_url': '',
                                    'height': 1024,
                                    'width': 2048,
                                    'date_captured': '2019-08-20 09:11:52.357475',
                                    'flickr_url': '',
                                    'id': image_id}

                    if len(kd['imgs'][img]['objects']) > 0:
                        for obj in kd['imgs'][img]['objects']:
                            if obj['category'] == "2":
                                category_id = 1
                            elif obj['category'] == "1":
                                category_id = 2
                            elif obj['category'] == "3":
                                category_id = 3
                            else:
                                category_id = 0

                            if category_id > 0:
                                xmin = max(int(obj['bbox']['xmin'])/2, 0)
                                ymin = max(int(obj['bbox']['ymin'])/2, 0)
                                xmax = min(int(obj['bbox']['xmax'])/2, w/2)
                                ymax = min(int(obj['bbox']['ymax'])/2, h/2)
                                len_w = xmax - xmin
                                len_h = ymax - ymin
                                if len_w > 15 and len_h > 15:
                                    annotation_l = {}
                                    if xmin < 2048:
                                        xmax = min(xmax, 2048)
                                        if xmax-xmin > 15:
                                            valid_l = True
                                            counter += 1
                                            annotation_l = {'segmentation': [[xmin, ymin], [xmax, ymin],
                                                                             [xmax, ymax], [xmin, ymax]],
                                                            'area': len_w*len_h,
                                                            'iscrowd': 0,
                                                            'image_id': image_l_id,
                                                            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                                                            'category_id': category_id,
                                                            'id': counter}
                                            train_annotations.append(annotation_l)

                                    annotation_r = {}
                                    if xmax > w/2 - 2048:
                                        r_xmin = max(xmin - (w/2 - 2048), 0)
                                        r_xmax = xmax - (w/2 - 2048)
                                        if r_xmax-r_xmin > 15:
                                            valid_r = True
                                            counter += 1
                                            annotation_r = {'segmentation': [[r_xmin, ymin], [r_xmax, ymin],
                                                                             [r_xmax, ymax], [r_xmin, ymax]],
                                                            'area': len_w*len_h,
                                                            'iscrowd': 0,
                                                            'image_id': image_r_id,
                                                            'bbox': [r_xmin, ymin, r_xmax - r_xmin, ymax - ymin],
                                                            'category_id': category_id,
                                                            'id': counter}
                                            train_annotations.append(annotation_r)

                                    annotation_m = {}
                                    if (xmax > (w/2 - 2048)/2 and xmax < (w/2 + 2048)/2) or \
                                            (xmin > (w/2 - 2048)/2 and xmin < (w/2 + 2048)/2):
                                        m_xmin = max(xmin - (w/2 - 2048)/2, 0)
                                        m_xmax = min(xmax - (w/2 - 2048)/2, 2048)
                                        if m_xmax-m_xmin > 15:
                                            valid_m = True
                                            counter += 1
                                            annotation_m = {'segmentation': [[m_xmin, ymin], [m_xmax, ymin],
                                                                             [m_xmax, ymax], [m_xmin, ymax]],
                                                            'area': len_w*len_h,
                                                            'iscrowd': 0,
                                                            'image_id': image_m_id,
                                                            'bbox': [m_xmin, ymin, m_xmax - m_xmin, ymax - ymin],
                                                            'category_id': category_id,
                                                            'id': counter}
                                            train_annotations.append(annotation_m)


                    if valid_l:
                        train_images.append(image_info_l)
                        crop_l_resize = cv2.resize(crop_l, dsize=(2048, 1024))
                        cv2.imwrite(os.path.join(
                            save_path, str(0) * 7 + str(image_l_id)+'.jpg'), crop_l_resize)
                    if valid_r:
                        train_images.append(image_info_r)
                        crop_r_resize = cv2.resize(crop_r, dsize=(2048, 1024))
                        cv2.imwrite(os.path.join(
                            save_path, str(0) * 7 + str(image_r_id)+'.jpg'), crop_r_resize)
                    if valid_m:
                        train_images.append(image_info_m)
                        crop_m_resize = cv2.resize(crop_m, dsize=(2048, 1024))
                        cv2.imwrite(os.path.join(
                            save_path, str(0) * 7 + str(image_m_id)+'.jpg'), crop_m_resize)


train_label_dict = {'info': info, 'images': train_images, 'licenses': licenses,
                    'annotations': train_annotations, 'categories': categories}


json_path = "G:\\impurity\\train_data\\20191204\\annotations"
if not os.path.exists(json_path):
    os.makedirs(json_path, exist_ok=True)

train_jsonfile = os.path.join(json_path, 'label.json')
with open(train_jsonfile, 'w') as json_file:
    json_file.write(json.dumps(train_label_dict, ensure_ascii=False))
    json_file.close()