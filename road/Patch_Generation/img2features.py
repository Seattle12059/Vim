import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models
import torch.nn as nn

class PatchFeatureExtractor:
    def __init__(self, img_tensor, patch_size, stride, noIP=False, device='cuda'):
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(device)
        # 将输入张量移动到指定设备
        img_tensor = img_tensor.to(self.device)

        # 确保输入是 PyTorch 张量
        if not isinstance(img_tensor, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor of shape (B, C, H, W)")

        # 保存原始张量
        self.img_tensor = img_tensor

        # 如果 stride 不是元组或列表，转换为元组
        if type(stride) not in (tuple, list):
            stride = (stride, stride)

        # 处理 stride，确保有四层的 stride
        stride = np.array(stride).squeeze()
        if len(stride.shape) == 1:
            stride = np.expand_dims(stride, 0).repeat(4, axis=0)  # 改为 4 层

        # patch 的大小
        self.patch_size = patch_size
        # 每层图像的 stride
        self.stride = stride

        # 如果不使用图像金字塔（noIP=True），则只使用原始图像
        if noIP:
            self.img_list = [self.img_tensor]
        else:
            # 将图像缩放到 600x600、300x300 和 900x600
            self.img1 = F.interpolate(self.img_tensor, size=(600, 600), mode='bilinear', align_corners=False)
            self.img2 = F.interpolate(self.img_tensor, size=(300, 300), mode='bilinear', align_corners=False)
            self.img3 = F.interpolate(self.img_tensor, size=(600, 900), mode='bilinear', align_corners=False)
            # 图像列表，包含四种尺寸的图像
            self.img_list = [self.img_tensor, self.img1, self.img2, self.img3]

        # 初始化特征提取器并移动到指定设备
        self.feature_extractor = self._init_feature_extractor()

    def _init_feature_extractor(self):
        model = torchvision.models.efficientnet_b3(pretrained=True)
        feature_extractor = nn.Sequential(*list(model.features))

        # 将特征提取器移动到指定设备
        feature_extractor = feature_extractor.to(self.device)

        for param in feature_extractor.parameters():
            param.requires_grad = False

        return feature_extractor

    # 提取单个 patch
    def extract_patch(self, img, stride):
        # 使用 unfold 提取 patches
        patches = img.unfold(2, self.patch_size, stride[0]).unfold(3, self.patch_size, stride[1])
        # 调整形状为 (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        B, C, H, W = img.shape
        num_patches_h = patches.size(2)
        num_patches_w = patches.size(3)
        # 将 patches 转换为 (B, num_patches, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C, self.patch_size, self.patch_size)
        return patches

    # 提取所有 patches
    def extract_patches(self):
        all_patches = []
        # 遍历每层图像及其对应的 stride
        for im, stride in zip(self.img_list, self.stride):
            # 提取当前层图像的所有 patches
            patches = self.extract_patch(im, stride)
            all_patches.append(patches)

        # 将所有 patches 拼接在一起，形状为 (B, total_patches, C, patch_size, patch_size)
        all_patches = torch.cat(all_patches, dim=1)
        return all_patches

    def extract_patch_features(self, patches):
        B, num_patches, C, H, W = patches.shape

        patches_reshaped = patches.view(B * num_patches, C, H, W)

        with torch.no_grad():
            features = self.feature_extractor(patches_reshaped)

        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))

        features = features.view(B, num_patches, -1)
        return features

    def process_patches(self):
        patches = self.extract_patches()
        patches_features = self.extract_patch_features(patches)
        return patches_features

# 示例用法
if __name__ == "__main__":
    # 创建一个随机的 (B, C, H, W) 张量
    img_tensor = torch.randn(4, 3, 1200, 900)
    patch_size = 300
    stride = 100

    # 创建 PatchExtractor 实例，默认使用可用的 GPU
    extractor = PatchFeatureExtractor(img_tensor, patch_size, stride)

    # 提取 patches
    patches = extractor.extract_patches()
    patches2 = extractor.process_patches()

    # 输出 patches 的形状
    print(patches.shape)
    print(patches2.shape)
