import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F


def generate_homography_low_level(size,
                                  TL=[0, 0],
                                  TR=[0, 0],
                                  BL=[0, 0],
                                  BR=[0, 0]):
    """使用opencv生成Homography矩阵

    Args:
        size (_type_): 这里的size应该是[height, width], 和TL,TR,BL,BR的定义有关
        TL (list, optional): . Defaults to [0, 0].
        TR (list, optional): . Defaults to [0, 0].
        BL (list, optional): . Defaults to [0, 0].
        BR (list, optional): . Defaults to [0, 0].

    Returns:
        _type_: _description_
    """
    h = size[0]
    w = size[1]

    p1 = np.array([[0.25, 0.25],   # Top-left
                   [0.25, 0.75],   # Top-right
                   [0.75, 0.25],   # Bottom-left
                   [0.75, 0.75]])  # Bootom-right

    p2 = np.array([[TL[0], TL[1]],
                   [TR[0], TR[1]],
                   [BL[0], BL[1]],
                   [BR[0], BR[1]]])

    p1[:, 0] = p1[:, 0] * h
    p1[:, 1] = p1[:, 1] * w

    p2[:, 0] = p2[:, 0] * h
    p2[:, 1] = p2[:, 1] * w

    H, _ = cv.findHomography(p1, p2)
    return H


def generate_random_homography(size, scale_en=True, translate_en=True, rotaion_en=True, perspective_en=True,
                               scale_range=[-0.1, 0.1],
                               translate_range=[-0.1, 0.1],
                               rotation_range=[-np.pi/3, np.pi/3],
                               perspective_range=[0, 0.01]):
    """_summary_

    Args:
        size (_type_): 需要注意, 这里的size的格式是[height, width]
        scale_en (bool, optional): . Defaults to True.
        translate_en (bool, optional): . Defaults to True.
        rotaion_en (bool, optional): . Defaults to True.
        perspective_en (bool, optional): . Defaults to True.
        scale_range (list, optional): . Defaults to [-0.15, 0.15].
        translate_range (list, optional): . Defaults to [-0.15, 0.15].
        rotation_range (list, optional): . Defaults to [-np.pi/2, np.pi/2].
        perspective_range (list, optional): . Defaults to [-0.01, 0.01].

    Returns:
        _type_: H矩阵
    """

    # 分别对应：TL TR BL BR
    p = np.array([[0.25, 0.25, 0.75, 0.75],
                  [0.25, 0.75, 0.25, 0.75],
                  [1.00, 1.00, 1.00, 1.00]])

    if scale_en:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        # TL
        p[0, 0] -= scale
        p[1, 0] -= scale
        # TR
        p[0, 1] -= scale
        p[1, 1] += scale
        # BL
        p[0, 2] += scale
        p[1, 2] -= scale
        # BR
        p[0, 3] += scale
        p[1, 3] += scale

    if translate_en:
        transX = np.random.uniform(translate_range[0], translate_range[1])
        transY = np.random.uniform(translate_range[0], translate_range[1])
        p[0, 0] += transX
        p[1, 0] += transY
        p[0, 1] += transX
        p[1, 1] += transY
        p[0, 2] += transX
        p[1, 2] += transY
        p[0, 3] += transX
        p[1, 3] += transY

    if rotaion_en:
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        translate_matrix = np.array([
            [1, 0, -0.5],
            [0, 1, -0.5],
            [0, 0, 1]
        ])

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        inverse_translate_matrix = np.array([
            [1, 0, 0.5],
            [0, 1, 0.5],
            [0, 0, 1]
        ])

        # 先平移到原点，再旋转
        p = translate_matrix.dot(p)
        p = rotation_matrix.dot(p)
        p = inverse_translate_matrix.dot(p)

    if perspective_en:
        perspective_bias = np.random.uniform(perspective_range[0], perspective_range[1])
        perspective_type = np.random.randint(0, 4)
        if perspective_type == 0:
            # Top, Horizental
            p[1, 0] += perspective_bias
            p[1, 1] -= perspective_bias
            # Top, Vertical
            # p[0, 0] -= perspective_bias
            # p[0, 1] -= perspective_bias
        elif perspective_type == 1:
            # Bottom, Horizental
            p[1, 2] += perspective_bias
            p[1, 3] -= perspective_bias
            # Bottom, Vertical
            # p[0, 2] += perspective_bias
            # p[0, 3] += perspective_bias
        elif perspective_type == 2:
            # Left, Horizental
            # p[1, 0] -= perspective_bias
            # p[1, 2] -= perspective_bias
            # Left, Vertical
            p[0, 0] += perspective_bias
            p[0, 2] -= perspective_bias
        elif perspective_type == 3:
            # Left, Horizental
            # p[1, 1] += perspective_bias
            # p[1, 3] += perspective_bias
            # Left, Vertical
            p[0, 1] += perspective_bias
            p[0, 3] -= perspective_bias

    TL = list(p[0:2, 0])
    TR = list(p[0:2, 1])
    BL = list(p[0:2, 2])
    BR = list(p[0:2, 3])

    return generate_homography_low_level(size, TL, TR, BL, BR)


class RandomHomography:
    def __init__(self):
        self.scale_en = True
        self.translate_en = True
        self.rotation_en = True
        self.perspective_en = True
        self.scale_range = [-0.1, 0.1]
        self.translate_range = [-0.1, 0.1]
        self.rotation_range = [-np.pi/3, np.pi/3]
        self.perspective_range = [0, 0.05]

    def warp_image(self, img: torch.Tensor):
        """进行随机单应性变换

        Args:
            img (torch.Tensor): 图片必须为[1, 1, Height, Width]的torch.float32类型的灰度图像

        Returns:
            warped_img: 进行随机变换后的图片, [1, 1, H, W]
            point0: 原始图像的点的位置(就是meshgrid)
            point1: 变换后图像点的位置
            corr0: 原始图像的indices(将2D图片flatten成1D之后)
            corr1: 变换后图像的indices(将2D图片flatten成1D之后)
            其中corr0和corr1是一一对应的
        """
        height, width = img.shape[2:4]
        H = generate_random_homography((height, width),
                                       self.scale_en,
                                       self.translate_en,
                                       self.rotation_en,
                                       self.perspective_en,
                                       self.scale_range,
                                       self.translate_range,
                                       self.rotation_range,
                                       self.perspective_range)

        H = torch.from_numpy(H).to(torch.float32)
        invH = torch.inverse(H)

        # HxWx2
        point0 = torch.dstack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij"))
        point0 = point0.to(torch.float32)
        # Nx2
        point0 = point0.view((-1, 2))
        # Nx3
        point0 = torch.concat([point0, torch.ones((height * width, 1))], dim=1)
        # 3xN
        point0 = point0.permute((1, 0))
        # 3xN
        point1 = torch.matmul(H, point0)     # 点Homography变换
        grid = torch.matmul(invH, point0)    # 用于图像Homography变换的grid
        # 2xN
        point0 = point0[0:2, :]
        point1 = point1[0:2, :] / point1[2:, :]
        grid = grid[0:2, :] / grid[2:, :]
        # Nx2
        point0 = point0.permute((1, 0))
        point1 = point1.permute((1, 0))
        grid = grid.permute((1, 0))

        # 采样
        grid[:, 0] = grid[:, 0] / (height // 2) - 1
        grid[:, 1] = grid[:, 1] / (width // 2) - 1
        grid = torch.concat([grid[:, 1:2], grid[:, 0:1]], dim=1)
        grid = grid.view((1, height, width, 2))
        warped_img = F.grid_sample(img.cpu(), grid, padding_mode='zeros', mode='bilinear', align_corners=False).to(img.device)

        corr0 = (point0[:, 0] * width + point0[:, 1] % width).to(torch.int32)
        corr1 = (point1[:, 0] * width + point1[:, 1] % width).to(torch.int32)

        return warped_img, point0, point1, corr0, corr1
