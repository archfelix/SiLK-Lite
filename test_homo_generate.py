import torch
import cv2 as cv
import numpy as np

import silk.homography as homo
import silk.utils as utils

randomhomo = homo.RandomHomography()
randomhomo.scale_en = True
randomhomo.translate_en = True
randomhomo.rotation_en = True
randomhomo.perspective_en = True

img = cv.imread('img_test0.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (160, 120))

img_tensor = utils.img_to_tensor(img, normalization=False)
while True:
    sample_img, warped_img, point0, point1, corr0, corr1 = randomhomo.generate_corrspodence(img_tensor)
    sample_img = sample_img.squeeze().detach().to(torch.uint8).numpy()
    warped_img = warped_img.squeeze().detach().to(torch.uint8).numpy()

    # 我们需要计算出两种H矩阵
    # 1. H matrix which transform sample_img to warped_img
    # 2. H matrix which transform warped_img to sample_img
    src_point = torch.concat([point0[:, 0:1], point0[:, 1:2]], dim=1).numpy()
    dst_point = torch.concat([point1[:, 0:1], point1[:, 1:2]], dim=1).numpy()
    H, H_mask = cv.findHomography(srcPoints=src_point, dstPoints=dst_point,
                                  method=cv.RANSAC,
                                  ransacReprojThreshold=1.0,
                                  maxIters=10000)
    print("Transform sample_img to warped_img")
    print(H)

    H, H_mask = cv.findHomography(srcPoints=dst_point, dstPoints=src_point,
                                  method=cv.RANSAC,
                                  ransacReprojThreshold=1.0,
                                  maxIters=10000)

    print("Transform warped_img to sample_img")
    print(H)

    cv.imshow("img0", sample_img)
    cv.imshow("img1", warped_img)
    key = cv.waitKey(0)
    if key == ord('w'):
        cv.imwrite("img0.jpg", sample_img)
        cv.imwrite("img1.jpg", warped_img)
        quit()
    elif key == ord('q'):
        quit()
