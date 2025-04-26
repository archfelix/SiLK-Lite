import torch
import cv2 as cv
import numpy as np

import silk.homography as homo
import silk.utils as utils

"""
这个文件用于生成特定变换矩阵的图片,用于检验算法是否正确
"""

randomhomo = homo.RandomHomography()
randomhomo.scale_en = False
randomhomo.translate_en = True
randomhomo.rotation_en = False
randomhomo.perspective_en = False

img = cv.imread('img0.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (160, 120))

img_tensor = utils.img_to_tensor(img, normalization=False)
while True:
    sample_img, warped_img, point0, point1, corr0, corr1 = randomhomo.generate_corrspodence(img_tensor)
    sample_img = sample_img.squeeze().detach().to(torch.uint8).numpy()
    warped_img = warped_img.squeeze().detach().to(torch.uint8).numpy()

    for i in range(point0.shape[0]):
        _sample_img = np.copy(sample_img)
        _warped_img = np.copy(warped_img)
        y0, x0 = point0[i]
        y1, x1 = point1[i]
        cv.circle(_sample_img, (int(x0), int(y0)), 1, (255, -1))
        cv.circle(_warped_img, (int(x1), int(y1)), 1, (255, -1))
        cv.imshow('sample_img', _sample_img)
        cv.imshow('wraped_img', _warped_img)
        cv.waitKey(1)
