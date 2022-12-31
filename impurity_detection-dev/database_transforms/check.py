import cv2
import os
import json


def draw_images(json_path, image_path):
    with open(json_path) as f:
        json_data = json.load(f)
        for sample_dict in json_data['annotations']:
            image_name = str(0)*7 + str(sample_dict['image_id']) + '.jpg'
            bbox = sample_dict['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = xmin + int(bbox[2])
            ymax = ymin + int(bbox[3])
            if min(xmin, ymin) < 0 or xmax > 1836 or ymax > 768:
                print(str(sample_dict['image_id']))
            #img = cv2.imread(os.path.join(image_path, image_name))
            #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
            #cv2.imwrite(os.path.join(image_path, image_name), img)


if __name__ == '__main__':
    json_path = "/home/ubuntu/project/Roadfaults/road_disease_det.v1.0/annotations/train.json"
    image_path = "/home/ubuntu/project/Roadfaults/road_disease_train/"
    draw_images(json_path, image_path)
