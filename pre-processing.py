import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def invertImg(img, imgname, savepath):
    img = img[:, :, 0:3]
    img_invert = np.abs(255 - img).astype('float32')
    cv2.imwrite(savepath + "under_1/" + imgname, img_invert.astype("uint8"))
    print(imgname + "complete!")
    return img_invert


def gamma15(img, imgname, savepath):
    img_norm = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
    img_norm_gc = (img_norm ** 1.5).astype('float32')
    result = cv2.normalize(img_norm_gc, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(savepath + "over_1.5/" + imgname, result.astype("uint8"))
    print(imgname + "complete!")


def adaptiveGamma(img, imgname, savepath):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_h, img_s, img_v = cv2.split(img_hsv)
    v_mean = np.mean(img_v)
    gamma = np.zeros((img.shape[0], img.shape[1]))
    A1 = 9.6
    A2 = 0.5
    x0 = 256 - 0.8 * v_mean
    p = 3.4

    gamma = A2 + (A1 - A2) / (1 + ((255 - img_v) / x0) ** p)

    gamma_val = np.zeros((img.shape[0], img.shape[1], 3))
    gamma_val[:, :, 0] = gamma
    gamma_val[:, :, 1] = gamma
    gamma_val[:, :, 2] = gamma
    img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    result = np.power(img, gamma_val).astype('float32')
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(savepath + "over_gc/" + imgname, result.astype("uint8"))
    print(imgname + "complete!")


def cpahe(img, imgname, savepath):
    img_pre_cpahe = img
    [W, H, C] = img_pre_cpahe.shape

    tmp_img_cpahe = np.zeros((W, H * 3))

    tmp_img_cpahe[:, 0::3] = img_pre_cpahe[:, :, 0]
    tmp_img_cpahe[:, 1::3] = img_pre_cpahe[:, :, 1]
    tmp_img_cpahe[:, 2::3] = img_pre_cpahe[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))

    tmp_img_cpahe = np.array(tmp_img_cpahe, dtype='uint8')
    img_cpahe_out = clahe.apply(tmp_img_cpahe)

    img_cpahe = np.zeros((W, H, 3))
    img_cpahe[:, :, 0] = img_cpahe_out[:, 0::3]
    img_cpahe[:, :, 1] = img_cpahe_out[:, 1::3]
    img_cpahe[:, :, 2] = img_cpahe_out[:, 2::3]

    cv2.imwrite(savepath + "over_cpahe/" + imgname, img_cpahe.astype("uint8"))
    print(imgname + "complete!")


if __name__ == '__main__':
    img_path = "dataset/test_data/"
    save_path = "pre_processing_test/"
    if os.path.exists(save_path + "under_1/") == False:
        os.mkdir(save_path + "under_1/")
        os.mkdir(save_path + "over_1.5/")
        os.mkdir(save_path + "over_gc/")
        os.mkdir(save_path + "over_cpahe/")

    for root, dirs, files in os.walk(img_path):
        for imgname in files:
            img_array = np.asarray(Image.open(os.path.join(root, imgname)))
            img_array = img_array[:, :, 0:3]

            try:
                invert_img = invertImg(img_array, imgname, save_path)
                gamma15(invert_img, imgname, save_path)
                adaptiveGamma(invert_img, imgname, save_path)
                cpahe(invert_img, imgname, save_path)

            except:
                print(imgname + " Error!")
