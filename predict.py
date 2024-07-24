
import time
import cv2
import numpy as np
from PIL import Image

from TransUNet2 import TransUNet2_Segmentation

if __name__ == "__main__":

    TransUNet2 = TransUNet2_Segmentation()

    mode = "dir_predict"

    count           = False

    name_classes    = ["background","landslides"]

    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = TransUNet2.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = TransUNet2.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
