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


def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=100):
    height, width = ktps.shape[2:4]
    kpts_map = torch.sigmoid(ktps).reshape(height*width)
    desc_map = desc.reshape((-1, height*width))
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice


def match_two(img0: torch.Tensor, img1: torch.Tensor):
    """this function can match two images.

    Args:
        img0 (torch.Tensor): shaped of [1, 1, H, W], formed by `utils.img_to_tensor`
        img1 (torch.Tensor): shaped of [1, 1, H, W], formed by `utils.img_to_tensor`
    """
    height, width = img0.shape[2:4]
    K = 100
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
        if sim_max[i] < 0.7:
            continue
        j = sim_indice[i].item()
        hw_pairs.append([
            int(sim_max[i] * 1000),
            topk_indice0[i] // width,
            topk_indice0[i] % width,
            topk_indice1[j] // width,
            topk_indice1[j] % width,
        ])
    return torch.tensor(hw_pairs, dtype=torch.int32)


IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120

img0 = cv.imread('img0.jpg')
img0 = cv.resize(img0, (IMAGE_WIDTH, IMAGE_HEIGHT))
img0_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
img0_tensor = silk.utils.img_to_tensor(img0_gray, device=device, normalization=True)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ok, img1 = cap.read()
    if not ok:
        break
    img1 = cv.resize(img1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img1_tensor = silk.utils.img_to_tensor(img1_gray, device=device, normalization=True)
    hw_pairs = match_two(img0=img0_tensor, img1=img1_tensor)
    img = np.hstack([img0, img1])

    for score, h0, w0, h1, w1 in hw_pairs:
        cv.line(img,
                (w0.item(), h0.item()),
                (IMAGE_WIDTH + w1.item(), h1.item()),
                color=(0, 255, 0),
                thickness=1,
                lineType=cv.LINE_4)
    cv.imshow('main', img)
    key = cv.waitKey(1)
