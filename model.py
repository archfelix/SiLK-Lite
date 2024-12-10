import torch
import torch.nn as nn
import torch.functional as F


class SiLK(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.conv_head = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.head_keypoint_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.head_keypoint_2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

        self.head_descriptor_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.head_descriptor_2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # backbone
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        # head
        x = self.act(self.conv_head(x))
        # keypoint head
        keypoint = self.act(self.head_keypoint_1(x))
        keypoint = self.head_keypoint_2(keypoint)
        # descriptor head
        descriptor = self.act(self.head_descriptor_1(x))
        descriptor = self.head_descriptor_2(descriptor)

        return keypoint, descriptor
