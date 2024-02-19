import cv2
import copy
import numpy as np
import torch


# 左上角低频矩阵，进行DCT逆变换
def low_frequency_idct(dct_clean, dct, dct_size):

    dct[dct_size + 1:, :] = dct_clean[dct_size + 1:, :]
    dct[:, dct_size + 1:] = dct_clean[:, dct_size + 1:]

    img = cv2.idct(dct)
    return img


def hight_frequency_idct(dct_clean,dct, dct_size):

    dct[:dct_size, :] = dct_clean[:dct_size, :]
    dct[:, :dct_size] = dct_clean[:, :dct_size]

    img = cv2.idct(dct)

    return img


def work(img_clean,img, img_size, dct_size,data_name):

    if data_name=='mnist':
        img_clean = np.float32(img_clean)
        img = np.float32(img)
        clean_dct = cv2.dct(img_clean)
        dct = cv2.dct(img)
        img = hight_frequency_idct(copy.deepcopy(clean_dct), copy.deepcopy(dct), img_size - dct_size)
        return img

    img_clean = np.float32(img_clean)
    r_clean, g_clean, b_clean = cv2.split(img_clean)
    img = np.float32(img)
    r, g, b = cv2.split(img)

    r_clean_dct = cv2.dct(r_clean)
    g_clean_dct = cv2.dct(g_clean)
    b_clean_dct = cv2.dct(b_clean)
    r_dct = cv2.dct(r)
    g_dct = cv2.dct(g)
    b_dct = cv2.dct(b)

    # r_img = low_frequency_idct(copy.deepcopy(r_clean_dct), copy.deepcopy(r_dct), dct_size)
    # g_img = low_frequency_idct(copy.deepcopy(g_clean_dct), copy.deepcopy(g_dct), dct_size)
    # b_img = low_frequency_idct(copy.deepcopy(b_clean_dct), copy.deepcopy(b_dct), dct_size)
    r_img = hight_frequency_idct(copy.deepcopy(r_clean_dct),copy.deepcopy(r_dct), img_size-dct_size)
    g_img = hight_frequency_idct(copy.deepcopy(g_clean_dct),copy.deepcopy(g_dct), img_size-dct_size)
    b_img = hight_frequency_idct(copy.deepcopy(b_clean_dct),copy.deepcopy(b_dct), img_size-dct_size)

    img = cv2.merge([r_img, g_img, b_img])
    return  img



def DCT(img_clean,img, img_size=28, dct_size=10,data_name=False):
    for i in range(img.shape[0]):
        img_clean1 = img_clean.data[i].cpu().numpy().transpose(1, 2, 0)
        img1 = img.data[i].cpu().numpy().transpose(1, 2, 0)
        change=work(copy.deepcopy(img_clean1),copy.deepcopy(img1), img_size, dct_size,data_name)
        if data_name=='mnist':
            change = np.expand_dims(change, axis=-1).transpose(2,0,1)
        else:
            change = change.transpose(2,0,1)
        img.data[i] = torch.from_numpy(change)
    return img
        
