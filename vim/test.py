import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np

class PatchExtractor:
    def __init__(self, img, patch_size, stride, noIP=False):
        # 如果输入是 numpy 数组，转换为 PIL 图像
        if type(img) == np.ndarray:
            img = Image.fromarray(img)

        # 如果 stride 不是元组或列表，转换为元组
        if type(stride) not in (tuple, list):
            stride = (stride, stride)

        # 处理 stride，确保有四层的 stride
        stride = np.array(stride).squeeze()
        if len(stride.shape) == 1:
            stride = np.expand_dims(stride, 0).repeat(4, axis=0)  # 改为 4 层

        # 原始图像
        self.img0 = img

        # 如果不使用图像金字塔（noIP=True），则只使用原始图像
        if noIP:
            self.img_list = [self.img0]
        else:
            # 将图像缩放到 600x600、300x300 和 900x600
            self.img1 = img.resize((600, 600), Image.BILINEAR)
            self.img2 = img.resize((300, 300), Image.BILINEAR)
            self.img3 = img.resize((900, 600), Image.BILINEAR)
            # 图像列表，包含四种尺寸的图像
            self.img_list = [self.img0, self.img1, self.img2, self.img3]

        # patch 的大小
        self.size = patch_size
        # 每层图像的 stride
        self.stride = stride

    # 提取单个 patch
    def extract_patch(self, img, patch, stride):
        # 使用 PIL 的 crop 方法提取 patch
        return img.crop((
            patch[0] * stride[0],  # 左边界
            patch[1] * stride[1],  # 上边界
            patch[0] * stride[0] + self.size,  # 右边界
            patch[1] * stride[1] + self.size  # 下边界
        ))

    # 计算当前层图像的 patch 数量
    def shape(self, img, stride):
        # 计算宽度方向的 patch 数量
        wp = int((img.width - self.size) / stride[1] + 1)
        # 计算高度方向的 patch 数量
        hp = int((img.height - self.size) / stride[0] + 1)
        return wp, hp

    # 提取所有 patches
    def extract_patches(self):
        patches = []
        # 遍历每层图像及其对应的 stride
        for im, stride in zip(self.img_list, self.stride):
            # 计算当前层图像的 patch 数量
            wp, hp = self.shape(im, stride)
            # 提取当前层图像的所有 patches
            temp = [self.extract_patch(im, (w, h), stride) for h in range(hp) for w in range(wp)]
            patches.extend(temp)
        return patches


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding with EfficientNetB3
    """

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.flatten = flatten

        # Initialize EfficientNetB3 for feature extraction
        self.efficientnet = models.efficientnet_b3(pretrained=True)
        self.efficientnet.classifier = nn.Identity()  # Remove the final classification layer

        # Projection layer to match the desired embed_dim
        self.proj = nn.Linear(1536, embed_dim)  # EfficientNetB3 output is 1536-dimensional
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # Convert tensor to PIL images for patch extraction
        patches = []
        for img in x:
            img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            patch_extractor = PatchExtractor(img_pil, self.patch_size, self.stride, noIP=True)
            patches.append(patch_extractor.extract_patches())

        # Convert patches to tensor and extract features using EfficientNetB3
        patch_features = []
        for batch_patches in patches:
            batch_features = []
            for patch in batch_patches:
                patch_tensor = torch.tensor(np.array(patch) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to CHW format
                patch_tensor = patch_tensor.to(x.device)
                features = self.efficientnet(patch_tensor)
                features = self.proj(features)
                batch_features.append(features)
            patch_features.append(torch.stack(batch_features))

        # Stack all batch features
        patch_features = torch.stack(patch_features)  # Shape: (B, N of patches, m dims of features)

        if self.flatten:
            patch_features = patch_features.flatten(2).transpose(1, 2)  # BCHW -> BNC

        patch_features = self.norm(patch_features)
        return patch_features


if __name__ == "__main__":
    # Example usage
    # Create a random input tensor (batch size = 2, channels = 3, height = 224, width = 224)
    x = torch.randn(2, 3, 224, 224)

    # Initialize the PatchEmbed model
    model = PatchEmbed(img_size=300, patch_size=16, stride=16, embed_dim=768)

    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)  # Expected shape: (2, N of patches, 768)