import cv2
import os
import copy


def mirror_annotation_enhancement(images_path, save_path_JPEG):
    for file in os.listdir(images_path):
        img = cv2.imread(os.path.join(images_path, file))
        num_img = str(0)*7 + str(int(file[-9:-4])+1)+'.jpg'
        print(num_img)
        print(img.shape)
        w = img.shape[1]
        h = img.shape[0]
        if w != 1630 and h != 1100:
            print(file + 'shape error.', w, h)
        mirror_img = copy.deepcopy(img)
        for wi in range(w):
            mirror_img[:, w-wi-1] = img[:, wi]
        cv2.imwrite(os.path.join(save_path_JPEG, num_img), mirror_img)


if __name__ == '__main__':
    images_path = "/data/deeplearning/wdq/high_speed_single_cls.v2/train/"
    save_path_JPEG = "/data/deeplearning/wdq/high_speed_single_cls.v2/train_m/"
    mirror_annotation_enhancement(images_path, save_path_JPEG)
