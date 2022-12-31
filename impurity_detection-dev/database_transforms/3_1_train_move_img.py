import cv2
import os

def move_images(train_txt_path, save_path):
    counter = 0
    if os.path.exists(save_path):
        os.removedirs(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for line in open(train_txt_path, "r"):
        if counter % 1 == 0:
            if line:
                split_res = os.path.split(line[:-1])
                print(split_res, split_res[1][:-9])
                img = cv2.imread(os.path.join(split_res[0], split_res[1][:-9]+'.jpg'))
                new_img = cv2.resize(img, (1630, 1100))
                w = new_img.shape[1]
                h = new_img.shape[0]
                if w != 1630 and h != 1100:
                    print(split_res[1][:-9] + 'shape error.', w, h)
                print(new_img.shape)
                cv2.imwrite(os.path.join(save_path, str(
                    0)*7 + split_res[1][-9:]), new_img)


if __name__ == '__main__':
    txt_path = "/data/deeplearning/wdq/high_speed_single_cls.v2/json/train.txt"
    save_path = "/data/deeplearning/wdq/high_speed_single_cls.v2/train/"
    move_images(txt_path, save_path)
