import cv2
import os
import json

def draw_images(json_path, image_path, dest_path = ""):

    if dest_path == "":
        dest_path = image_path
    else:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

    with open(json_path) as f:
        json_data = json.load(f)
        for sample_dict in json_data['annotations']:
            image_name = str(0)*7 + str(sample_dict['image_id']) + '.jpg'
            bbox = sample_dict['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = xmin + int(bbox[2])
            ymax = ymin + int(bbox[3])
            img = cv2.imread(os.path.join(image_path, image_name))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(dest_path, image_name), img)

if __name__ == '__main__':
    json_path = 'G:\\impurity\\train_data\\20191204\\annotations\\label.json'
    image_path = "G:\\impurity\\train_data\\20191204"
    dest_path = 'G:\\impurity\\train_data\\20191204\\img_preview'
    draw_images(json_path, image_path, dest_path)
