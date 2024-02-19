import cv2
import copy
import numpy as np
import torch


def low_frequency_idct(dct, dct_size):

    dct[dct_size + 1:, :] = 0
    dct[:, dct_size + 1:] = 0

    img = cv2.idct(dct)
    return img

def hight_frequency_idct(dct, dct_size):

    dct[:dct_size, :] = 0
    dct[:, :dct_size ] = 0

    img = cv2.idct(dct)

    return img

def work(img, img_size, dct_size):

    img = np.float32(img)
    r, g, b = cv2.split(img)

    r_dct = cv2.dct(r)
    g_dct = cv2.dct(g)
    b_dct = cv2.dct(b)

    r_img = hight_frequency_idct(copy.deepcopy(r_dct), img_size-dct_size)
    g_img = hight_frequency_idct(copy.deepcopy(g_dct), img_size-dct_size)
    b_img = hight_frequency_idct(copy.deepcopy(b_dct), img_size-dct_size)
    # r_img = low_frequency_idct(copy.deepcopy(r_dct), dct_size)
    # g_img = low_frequency_idct(copy.deepcopy(g_dct), dct_size)
    # b_img = low_frequency_idct(copy.deepcopy(b_dct), dct_size)

    img = cv2.merge([r_img, g_img, b_img])
    return  img


def DCT(img, img_size=28, dct_size=10):
    for i in range(img.shape[0]):
        img1 = img.data[i].cpu().numpy().transpose(1, 2, 0)
        change=work(copy.deepcopy(img1), img_size, dct_size)
        change = change.transpose(2,0,1)
        img.data[i] = torch.from_numpy(change)
    return img
        
