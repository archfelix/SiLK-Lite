import torch
import torch.nn as nn
import torch.functional as F


# class SiLK(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.relu = nn.ReLU(inplace=True)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)

#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)

#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)

#         self.conv_head = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(128)

#         self.head_keypoint_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn_keypoint_1 = nn.BatchNorm2d(128)
#         self.head_keypoint_2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

#         self.head_descriptor_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn_descriptor_1 = nn.BatchNorm2d(128)
#         self.head_descriptor_2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         # backbone
#         x = self.relu(self.conv1(x))
#         x = self.bn1(x)

#         x = self.relu(self.conv2(x))
#         x = self.bn2(x)

#         x = self.relu(self.conv3(x))
#         x = self.bn3(x)

#         x = self.relu(self.conv4(x))
#         x = self.bn4(x)

#         x = self.relu(self.conv5(x))
#         x = self.bn5(x)

#         # head
#         x = self.relu(self.conv_head(x))
#         x = self.bn6(x)
#         # keypoint head
#         keypoint = self.relu(self.head_keypoint_1(x))
#         keypoint = self.bn_keypoint_1(keypoint)
#         keypoint = self.head_keypoint_2(keypoint)
#         # descriptor head
#         descriptor = self.relu(self.head_descriptor_1(x))
#         descriptor = self.bn_descriptor_1(descriptor)
#         descriptor = self.head_descriptor_2(descriptor)

#         return keypoint, descriptor


class SiLK(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv_head = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn_head = nn.BatchNorm2d(128)

        self.head_keypoint_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn_keypoint_1 = nn.BatchNorm2d(128)
        self.head_keypoint_2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

        self.head_descriptor_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn_descriptor_1 = nn.BatchNorm2d(128)
        self.head_descriptor_2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # backbone
        x = self.relu(self.conv1(x))
        x = self.bn1(x)

        x = self.relu(self.conv2(x))
        x = self.bn2(x)

        x = self.relu(self.conv3(x))
        x = self.bn3(x)

        # head
        x = self.relu(self.conv_head(x))
        x = self.bn_head(x)
        # keypoint head
        keypoint = self.relu(self.head_keypoint_1(x))
        keypoint = self.bn_keypoint_1(keypoint)
        keypoint = self.head_keypoint_2(keypoint)
        # descriptor head
        descriptor = self.relu(self.head_descriptor_1(x))
        descriptor = self.bn_descriptor_1(descriptor)
        descriptor = self.head_descriptor_2(descriptor)

        return keypoint, descriptor
