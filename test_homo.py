import homography as homo
import torch
import cv2 as cv
import utils

randomhomo = homo.RandomHomography()
randomhomo.scale_en = True
randomhomo.translate_en = True
randomhomo.rotation_en = True
randomhomo.perspective_en = True

img = cv.imread('test.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (40, 30))

img_tensor = utils.img_to_tensor(img)
while True:
    wrapped_img, point0, point1, corr0, corr1 = randomhomo.warp_image(img_tensor)
    wrapped_img = wrapped_img.squeeze().detach().to(torch.uint8).numpy()

    cv.imshow('img', img)
    cv.imshow('wrapped_img', wrapped_img)
    cv.waitKey(0)

    # for y, x in point0:
    #     cv.circle(img, (int(x), int(y)), 1, 255, 1)

    # for y, x in point1:
    #     cv.circle(wrapped_img, (int(x), int(y)), 1, 255, 1)

    # cv.imshow('img', img)
    # cv.imshow('wrapped_img', wrapped_img)
    # cv.waitKey(0)