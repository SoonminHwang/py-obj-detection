import cv2
import numpy.random as npr
from fast_rcnn.config import cfg
import numpy as np

def _crop_resize(image, param):
    crop = image[param[0]:param[1], param[2]:param[3], ...]
    image_cr = cv2.resize(crop, (param[5], param[4]))
    return image_cr

def _gamma_correction(image):
    invGamma = 1.0 / npr.uniform(cfg.TRAIN.GAMMA_RNG[0], cfg.TRAIN.GAMMA_RNG[1], size=(1))
    table = np.array( [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

    image_g = cv2.LUT( image, table )

    return image_g

def _flip(image):
	return image[:, ::-1, ...]