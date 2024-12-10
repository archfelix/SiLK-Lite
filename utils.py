import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv

from model import SiLK
import homography as homo
import dyncosim

"""
def loss(image_0, model):
    # get warped image and pixel correspondences
    image_1, corr_0, corr_1 = rand_homo(image_0)
    
    # apply image augmentations
    image_0 = augment(image_0)
    image_1 = augment(image_1)
    
    # extract dense descriptors and keypoints
    desc_0, kpts_0 = model(image_0)
    desc_1, kpts_1 = model(image_1)
    
    # compute similarity matrix
    sim_mat = cosim(desc_0, desc_1)
    
    # compute the descriptor loss
    # using ground truth correspondences
    loss_desc = nll(sim_mat, corr_0, corr_1)
    
    # measure matching success
    y = is_match_success(sim_mat, corr_0, corr_1)
    
    # compute keypoint loss
    # using current matching success
    loss_kpts = bce(kpts_0, y, corr_0)
    loss_kpts += bce(kpts_1, y, corr_1)
    return loss_desc + loss_kpts
"""


def img_to_tensor(img: np.ndarray):
    """_summary_

    Args:
        img (np.ndarray): [Height, Width]
    """

    img_tensor = torch.from_numpy(img).to(torch.float32)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def tensor_to_img(img: torch.Tensor):
    return img.cpu().detach().squeeze().numpy()


random_homo = homo.RandomHomography()
random_homo.scale_en = True
random_homo.translate_en = True
random_homo.rotation_en = True
random_homo.perspective_en = False


def rand_homo(img):
    warped_img, _, _, corr0, corr1 = random_homo.warp_image(img=img)
    return warped_img, corr0, corr1


def filter_corr(corr0: torch.Tensor, corr1: torch.Tensor, H=0, W=0):
    mask = ((corr1 >= 0) & (corr1 < H*W))
    corr_tensor = torch.concat([corr0[mask].unsqueeze(1), corr1[mask].unsqueeze(1)], dim=1)
    return corr_tensor


def apply_augment(img):
    return img


"""
# Used for block-computation, but it is not necessary
def compute_nll_loss(sim_mat: dyncosim.DynamicCosineSim, corr_pos: torch.Tensor, block_size=100):
    loss = torch.zeros((1), device=sim_mat.device)
    N = corr_pos.shape[0]
    B = N // block_size
    Left = N % block_size
    # 分块计算
    for i in range(B):
        loss += torch.sum(torch.log(sim_mat.get_Pij(corr_pos[i*block_size:(i+1)*block_size, :])), 0)
    # 处理剩余
    if Left > 0:
        loss += torch.sum(torch.log(sim_mat.get_Pij(corr_pos[i*block_size:i*block_size + Left, :])), 0)
    loss = loss / -N
    return loss


def is_match_success(sim_mat: dyncosim.DynamicCosineSim, corr_pos: torch.Tensor, block_size=100, min_score=0.6):
    N = corr_pos.shape[0]
    y = torch.zeros((N), device=sim_mat.device)
    B = N // block_size
    Left = N % block_size
    # 分块计算
    for i in range(B):
        matching_score = sim_mat.get_Pij(corr_pos[i*block_size:(i+1)*block_size, :])
        mask = matching_score > min_score
        y[i*block_size:(i+1)*block_size][mask] = 1.0

    # 处理剩余
    if Left > 0:
        matching_score = sim_mat.get_Pij(corr_pos[i*block_size:i*block_size + Left, :])
        mask = matching_score > min_score
        y[i*block_size:i*block_size + Left][mask] = 1.0

    return y
"""


def is_match_success(sim_mat: dyncosim.DynamicCosineSim, corr_pos: torch.Tensor):
    N = corr_pos.shape[0]
    y = torch.zeros((N), device=sim_mat.device)
    sim_corr = sim_mat.get_sim_ij(corr_pos)
    mask = (sim_corr >= sim_mat.max_sik[corr_pos[:, 0]]) & (sim_corr >= sim_mat.max_skj[corr_pos[:, 1]])
    y[mask] = 1.0
    return y, mask[mask].shape[0]


def compute_nll_loss(sim_mat: dyncosim.DynamicCosineSim, corr_pos: torch.Tensor):
    N = corr_pos.shape[0]
    loss = torch.sum(sim_mat.get_Pij(corr_pos), dim=0) / -N
    return loss


def compute_kpt_loss(corr_tenor: torch.Tensor,
                     kpts0: torch.Tensor,
                     kpts1: torch.Tensor,
                     y: torch.Tensor):
    N = y.shape[0]

    B, C, H, W = kpts0.shape
    assert B == 1 and C == 1
    kpts0 = torch.maximum(torch.tensor(-70), kpts0.view(H*W))
    kpts1 = torch.maximum(torch.tensor(-70), kpts1.view(H*W))

    def bce(q, y, c):
        # 正例: q最终会大于0
        # 负例: q最终会小于0
        # 注意, 当q接近-100的时候，sigmoid会输出0, log则会导致inf
        return torch.sum(y*torch.log(F.sigmoid(q[c])) + (1-y) * torch.log(F.sigmoid(-q[c]))) / -N

    loss = bce(kpts0, y, corr_tenor[:, 0]) + bce(kpts1, y, corr_tenor[:, 1])
    return loss


def compute_loss(model: SiLK, img0, block_size=100, tau=1):
    img1, corr0, corr1 = rand_homo(img0)
    corr_tensor = filter_corr(corr0, corr1, H=img1.shape[2], W=img1.shape[3])

    img0 = apply_augment(img0)
    img1 = apply_augment(img1)

    kpts0, desc0 = model.forward(img0)
    kpts1, desc1 = model.forward(img1)

    sim_mat = dyncosim.DynamicCosineSim(desc0, desc1, block_size=block_size, tau=tau)

    loss_desc = compute_nll_loss(sim_mat, corr_tensor)

    y, count = is_match_success(sim_mat, corr_tensor)

    loss_kpts = compute_kpt_loss(corr_tensor, kpts0, kpts1,  y)

    kpts_img = torch.sigmoid(kpts0)
    img = tensor_to_img(kpts_img)
    img[img > 0.8] = 255
    cv.imshow('kpts0,', img.astype(np.uint8))

    return (loss_desc + loss_kpts,
            loss_desc.item(),
            loss_kpts.item(),
            count)
