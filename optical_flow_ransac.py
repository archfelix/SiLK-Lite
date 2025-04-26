import torch
import cv2 as cv
import numpy as np
import silk
import random
import math
import lpf
from silk.homography import RandomHomography

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


K = 100

lpf_x = lpf.LowpassFilter(alpha=0.5)
lpf_y = lpf.LowpassFilter(alpha=0.5)
lpf_theta = lpf.LowpassFilter(alpha=0.5)
homo = RandomHomography()

img0 = cv.imread('img0.jpg')
img1 = cv.imread('img1.jpg')

img0 = cv.resize(img0, (160, 120))
img1 = cv.resize(img1, (160, 120))

img0_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img0_tensor = silk.utils.img_to_tensor(img0_gray, device=device, normalization=True)
img1_tensor = silk.utils.img_to_tensor(img1_gray, device=device, normalization=True)

hw_pairs = match_two(img0=img0_tensor, img1=img1_tensor, K=K, sim_thres=0.7, kpts_thres=0.2)

# Use OpenCV to get Homography Matrix
# src_point = torch.concat([hw_pairs[:, 4:5], hw_pairs[:, 3:4]], dim=1).numpy()
# dst_point = torch.concat([hw_pairs[:, 2:3], hw_pairs[:, 1:2]], dim=1).numpy()
src_point = torch.concat([hw_pairs[:, 3:4], hw_pairs[:, 4:5]], dim=1).numpy()
dst_point = torch.concat([hw_pairs[:, 1:2], hw_pairs[:, 2:3]], dim=1).numpy()
H, H_mask = cv.findHomography(srcPoints=src_point, dstPoints=dst_point,
                              method=cv.RANSAC,
                              ransacReprojThreshold=1.0,
                              maxIters=1000)

"""
Transform sample_img to warped_img
[[ 8.10783689e-01 -3.90443926e-02  3.17044125e+01]
 [ 2.13909989e-01  1.00036411e+00 -6.22340444e-01]
 [-3.85993000e-04  1.38826723e-03  1.00000000e+00]]
 
Transform warped_img to sample_img
[[ 1.22400376e+00  1.02086724e-01 -3.88264127e+01]
 [-2.61172481e-01  1.00571611e+00  8.91139377e+00]
 [ 8.40813248e-04 -1.34904994e-03  1.00000000e+00]]
"""

print("Optical Flow: transform img1(warped_img) to img0(sample_img)")
print(H)
offset_x = lpf_x.update(H[0, 2] / H[2, 2])
offset_y = lpf_y.update(H[1, 2] / H[2, 2])
offset_theta = lpf_theta.update(math.atan2(H[1, 0], H[0, 0]))
print(f'X={offset_x:10.5f}  Y={offset_y:10.5f}   Theta={offset_theta:10.5f}')

warped_img = homo.warp_image(
    silk.utils.img_to_tensor(img1_gray, device=device, normalization=False),
    torch.tensor(H, dtype=torch.float32))

warped_img = warped_img.squeeze().cpu().to(torch.uint8).numpy()


cv.imshow("img0", img0)
cv.imshow("img1", img1)
cv.imshow("warped_img", warped_img)
cv.waitKey(0)
