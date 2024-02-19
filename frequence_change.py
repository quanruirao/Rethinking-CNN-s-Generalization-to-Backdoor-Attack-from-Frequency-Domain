import cv2
import copy
import numpy as np
import torch
from scipy.interpolate import interp2d


def hight_frequency_idct(dct_clean,dct, dct_size):

    dct[-dct_size:, -dct_size:] += dct_clean

    img = cv2.idct(dct)

    return img


def low_frequency_clip(dct_clean,dct_size,intensity=0.05):

    dct_low = dct_clean[:dct_size , :dct_size ]
    dct_low = (dct_low - np.min(dct_low)) / (np.max(dct_low)-np.min(dct_low))
    dct_low = intensity*dct_low

    x = np.arange(dct_low.shape[0])
    y = np.arange(dct_low.shape[1])

    interp_func = interp2d(x, y, dct_low, kind='linear')


    xi = np.linspace(0, 1, int(dct_clean.shape[1] / 2))
    yi = np.linspace(0, 1, int(dct_clean.shape[1] / 2))

    target_matrix = interp_func(xi, yi)


    return target_matrix



def work(img_clean,img, img_size, dct_size,intensity=0.05):

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


    r_clean_freq = low_frequency_clip(r_clean_dct, dct_size,intensity=intensity)
    g_clean_freq = low_frequency_clip(g_clean_dct, dct_size,intensity=intensity)
    b_clean_freq = low_frequency_clip(b_clean_dct, dct_size,intensity=intensity)

    r_img = hight_frequency_idct(copy.deepcopy(r_clean_freq),copy.deepcopy(r_dct), int(img.shape[1] / 2))
    g_img = hight_frequency_idct(copy.deepcopy(g_clean_freq),copy.deepcopy(g_dct), int(img.shape[1] / 2))
    b_img = hight_frequency_idct(copy.deepcopy(b_clean_freq),copy.deepcopy(b_dct), int(img.shape[1] / 2))

    img = cv2.merge([r_img, g_img, b_img])

    return  img

def work_mnist(img_clean,img, img_size, dct_size,intensity=0.05):

    img_clean = np.float32(img_clean)
    r_clean = img_clean[:, :, 0]
    img = np.float32(img)
    r = img[:, :, 0]


    r_clean_dct = cv2.dct(r_clean)
    r_dct = cv2.dct(r)

    r_clean_freq = low_frequency_clip(r_clean_dct, dct_size,intensity=intensity)

    r_img = hight_frequency_idct(copy.deepcopy(r_clean_freq),copy.deepcopy(r_dct), int(img.shape[1] / 2))


    img = np.expand_dims(r_img, axis=-1)

    return  img


def DCT(img_clean,img, img_size=28, dct_size=10,is_mnist=False,intensity=0.05):
    for i in range(img.shape[0]):
        img_clean1 = img_clean.numpy().transpose(1, 2, 0)
        img1 = img.data[i].cpu().numpy().transpose(1, 2, 0)

        if is_mnist:
            change = work_mnist(copy.deepcopy(img_clean1), copy.deepcopy(img1), img_size, dct_size,intensity=intensity)
        else:
            change=work(copy.deepcopy(img_clean1),copy.deepcopy(img1), img_size, dct_size,intensity=intensity)

        change = change.transpose(2,0,1)
        img.data[i] = torch.from_numpy(change)
    return img
        
