import cv2
import os
import json

def draw_images(json_path, image_path, save_path):
    with open(json_path) as f:
        json_data = json.load(f)
        image_dict = json_data['imgs']
        for k, v in image_dict.items():
            image = cv2.imread(os.path.join(image_path, k))
            if image is not None:
                bboxes = v['objects']
                for per_box in bboxes:
                    bbox = per_box["bbox"]
                    xmin = int(bbox["xmin"])
                    ymin = int(bbox["ymin"])
                    xmax = int(bbox["xmax"])
                    ymax = int(bbox["ymax"])
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
                cv2.imwrite(os.path.join(save_path, k), image)


if __name__ == '__main__':
    json_path = "/data/deeplearning/wdq/orthographic_datasets/204359490_20191006110809380/crack_detection.json"
    image_path = "/data/deeplearning/wdq/orthographic_datasets/204359490_20191006110809380/orthographic_coarse/"
    save_path = "/data/deeplearning/wdq/orthographic_datasets/204359490_20191006110809380/crack_detection_coarse"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    draw_images(json_path, image_path, save_path)
