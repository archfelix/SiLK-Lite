import torch
import cv2 as cv
import numpy as np
import silk
import random
import math
import lpf

import time

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


def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=100):
    height, width = ktps.shape[2:4]
    kpts_map = torch.sigmoid(ktps).reshape(height*width)
    desc_map = desc.reshape((-1, height*width))
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice


def match_two(img0: torch.Tensor, img1: torch.Tensor, K=100, sim_thres=0.8, kpts_thres=0.2):
    """this function can match two images.

    Args:
        img0 (torch.Tensor): shaped of [1, 1, H, W], formed by `utils.img_to_tensor`
        img1 (torch.Tensor): shaped of [1, 1, H, W], formed by `utils.img_to_tensor`
    Return:
        [[Score, img0 H index, img0 W index, img1 H index, img1 W index],...]
    """
    height, width = img0.shape[2:4]
    topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0), k=K)
    topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1), k=K)
    topk_desc0 = topk_desc0.permute(1, 0)  # shape = [K, 128]
    topk_desc1 = topk_desc1.permute(1, 0)  # shape = [K, 128]
    topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
    topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)
    sim_mat = torch.matmul(topk_desc0, topk_desc1.T)  # Similarity from desc0 to desc1
    sim_max, sim_indice = torch.max(sim_mat, dim=1)
    hw_pairs = []
    for i in range(K):
        j = sim_indice[i].item()
        if sim_max[i] < sim_thres:
            continue
        if topk_kpts0[i] < kpts_thres:
            continue
        if topk_kpts1[j] < kpts_thres:
            continue
        hw_pairs.append([
            int(sim_max[i] * 1000),
            topk_indice0[i] // width,  # img0
            topk_indice0[i] % width,  # img0
            topk_indice1[j] // width,  # img1
            topk_indice1[j] % width,  # img1
        ])
    return torch.tensor(hw_pairs, dtype=torch.int32)


def generate_n_int(low=0, high=100, n=4):
    """generate N random numbers, range in [low, high), not including both boundary.

    Args:
        low (int, optional): low boundary. Defaults to 0.
        high (int, optional): high boundary. Defaults to 100.
        n (int, optional): amount. Defaults to 4.
    """
    random_sample = set()
    while True:
        a = random.randint(low, high - 1)
        random_sample.add(a)
        if len(random_sample) == n:
            break
    return list(random_sample)


IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
K = 200
img0_tensor = None
lpf_x = lpf.LowpassFilter(alpha=0.8)
lpf_y = lpf.LowpassFilter(alpha=0.8)
lpf_theta = lpf.LowpassFilter(alpha=0.8)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)


def update(frame):
    global img0_tensor

    t0 = time.time()
    ok, img1 = cap.read()
    if not ok:
        return

    img1 = cv.resize(img1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    cv.imshow('main', img1)

    t1 = time.time()

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img1_tensor = silk.utils.img_to_tensor(img1_gray, device=device, normalization=True)
    if img0_tensor is None:
        img0_tensor = img1_tensor
        return

    hw_pairs = match_two(img0=img0_tensor, img1=img1_tensor, K=K, sim_thres=0.80, kpts_thres=0.2)
    if hw_pairs.shape[0] < 4:
        img0_tensor = img1_tensor
        return

    t2 = time.time()

    # Use OpenCV to get Homography Matrix
    src_point = torch.concat([hw_pairs[:, 4:5], hw_pairs[:, 3:4]], dim=1).numpy()
    dst_point = torch.concat([hw_pairs[:, 2:3], hw_pairs[:, 1:2]], dim=1).numpy()
    H, H_mask = cv.findHomography(srcPoints=src_point, dstPoints=dst_point,
                                  method=cv.RANSAC,
                                  ransacReprojThreshold=1.0,
                                  maxIters=1000)

    if H is not None:
        p0 = np.array([[IMAGE_WIDTH // 2, IMAGE_WIDTH - 1],
                       [IMAGE_HEIGHT // 2, IMAGE_HEIGHT // 2],
                       [1, 1]])
        p1 = np.dot(H, p0)
        p1 = p1 / p1[2, :]
        # 求平移
        dx = p1[0, 0] - p0[0, 0]
        dy = p1[1, 0] - p0[1, 0]
        # 求旋转角度
        v0 = p0[:, 1] - p0[:, 0]
        v1 = p1[:, 1] - p1[:, 0]

        cos_theta = (v0 @ v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
        sign = np.sign(v0[0] * v1[1] - v0[1]*v1[0])
        theta = np.arccos(cos_theta) * sign
        offset_x = lpf_x.update(dx)
        offset_y = lpf_y.update(dy)
        offset_theta = lpf_theta.update(theta)

        # print(f'X={offset_x:10.5f}  Y={offset_y:10.5f}   Theta={offset_theta:10.5f}')

    img0_tensor = img1_tensor
    t3 = time.time()

    print(f't1-t0={t1 - t0}  t2-t1={t2 - t1}  t3-t2={t3 - t2}  t3-t0={t3 - t0}')


frame_index = 0
while True:
    update(frame_index)
    cv.waitKey(1)
    frame_index += 1
