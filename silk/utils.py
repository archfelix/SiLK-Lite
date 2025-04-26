import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import albumentations as A

from .model import SiLK
from .homography import RandomHomography
from .dyncosim import DynamicCosineSim

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

silk_augmentation = A.Compose([
    A.Blur(
        p=0.1,
        blur_limit=(3, 3)
    ),
    A.MotionBlur(
        p=0.2,
        blur_limit=(3, 3)
    ),
    A.RandomBrightnessContrast(
        p=0.5,
        brightness_limit=(-0.1, 0),
        contrast_limit=(-0.1, 0.1)
    ),
    A.GaussNoise(
        p=0.5,
        std_range=(0.01, 0.03),
    )
], p=0.95)


def img_to_tensor(img: np.ndarray, device=torch.device("cpu"), normalization=False):
    """_summary_

    Args:
        img (np.ndarray): [Height, Width]
    """

    img_tensor = torch.from_numpy(img).to(torch.float32).to(device)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.unsqueeze(0)
    if normalization:
        img_tensor = img_tensor / 255.0
    return img_tensor


def tensor_to_img(img: torch.Tensor):
    return img.cpu().detach().squeeze().numpy()


def tensor_show(win, img: torch.Tensor):
    img = tensor_to_img(img * 255).astype(np.uint8)
    cv.imshow(win, img)


random_homo = RandomHomography()
random_homo.scale_en = True
random_homo.translate_en = True
random_homo.rotation_en = True
random_homo.perspective_en = True


def rand_homo(img):
    sampled_img, warped_img, _, _, corr0, corr1 = random_homo.generate_corrspodence(img=img)
    return sampled_img, warped_img, corr0, corr1


def apply_augment(img_tensor: torch.Tensor):
    """
    img=[1, 1, H, W]
    """
    img = tensor_to_img(img_tensor)
    result = silk_augmentation(image=img)
    return img_to_tensor(result['image'], device=img_tensor.device)


def is_match_success(sim_mat: DynamicCosineSim, corr_pos: torch.Tensor):
    N = corr_pos.shape[0]
    y = torch.zeros((N), device=sim_mat.device)
    sim_corr = sim_mat.get_sim_ij(corr_pos)
    mask = (sim_corr >= sim_mat.max_sik[corr_pos[:, 0]]) & (sim_corr >= sim_mat.max_skj[corr_pos[:, 1]])
    y[mask] = 1.0
    return y, mask[mask].shape[0]


def compute_nll_loss(sim_mat: DynamicCosineSim, corr_pos: torch.Tensor):
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
    kpts0 = kpts0.view(H*W)
    kpts1 = kpts1.view(H*W)

    def BCE(q, y, c):
        # return torch.sum(y*torch.log(F.sigmoid(q[c])) + (1-y) * torch.log(F.sigmoid(-q[c]))) / -N
        return F.binary_cross_entropy(input=F.sigmoid(q[c]), target=y, reduction='mean')

    loss = BCE(kpts0, y, corr_tenor[:, 0]) + BCE(kpts1, y, corr_tenor[:, 1])
    return loss


def compute_loss(model: SiLK, img0: torch.Tensor, tau=0.05, block_size=None):
    img0, img1, corr0, corr1 = rand_homo(img0)
    corr_tensor = torch.concat([corr0.unsqueeze(1), corr1.unsqueeze(1)], dim=1)

    img0 = apply_augment(img0)
    img1 = apply_augment(img1)

    # used to show the effect of augmentation
    # tensor_show('img0', img0)
    # tensor_show('img1', img1)

    kpts0, desc0 = model.forward(img0)
    kpts1, desc1 = model.forward(img1)

    sim_mat = DynamicCosineSim(desc0, desc1, block_size=block_size, tau=tau)

    loss_desc = compute_nll_loss(sim_mat, corr_tensor)

    y, count = is_match_success(sim_mat, corr_tensor)

    loss_kpts = compute_kpt_loss(corr_tensor, kpts0, kpts1,  y)

    return (loss_desc + loss_kpts,
            loss_desc.item(),
            loss_kpts.item(),
            count)
