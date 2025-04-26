import torch
import cv2 as cv
import numpy as np
import silk

device = None
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
    device = torch.device("cuda:0")
else:
    print("CUDA is not available. Training on CPU.")
    device = torch.device("cpu")

model = silk.SiLK()
model = model.to(device)
model.train(False)

model.load_state_dict(torch.load("./train0_30000.pth"))

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ok, img = cap.read()
    if not ok:
        break
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.resize(img, (320, 240))
    img = cv.resize(img, (160, 120))
    # img = cv.resize(img, (80, 60))
    # img = cv.GaussianBlur(img, (3, 3), sigmaX=1)
    # predict
    img_tensor = silk.utils.img_to_tensor(img, device=device, normalization=True)
    kpts, desc = model.forward(img_tensor)

    # show
    height, width = kpts.shape[2:4]
    kpts_img = torch.sigmoid(kpts).cpu().view(-1)
    topk_values, topk_indice = kpts_img.topk(500)
    for i in range(len(topk_indice)):
        value = topk_values[i]
        indice = topk_indice[i]
        if value < 0.26:
            continue
        h = indice // width
        w = indice % width
        cv.circle(img, (int(w), int(h)), 1, (255, -1))

    img = cv.resize(img, (width*4, height*4))
    cv.imshow('main', img)
    cv.waitKey(1)
