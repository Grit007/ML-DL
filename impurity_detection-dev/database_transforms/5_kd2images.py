import os
import json
import numpy as np
import cv2


def kd2images(path, save_path):
    counter = 10000
    for folder in os.walk(path):
        for file in folder[2]:
            file_path = folder[0]+'/'+file
            if file_path.endswith('.jpg'):
                counter += 1
                img = cv2.imread(file_path)
                cv2.imwrite(save_path + '/' + str(counter)+'.jpg', img)


kd2images('/home/ubuntu/project/Roadfaults/road_disease_db/',
          '/home/ubuntu/project/Roadfaults/total_images')
