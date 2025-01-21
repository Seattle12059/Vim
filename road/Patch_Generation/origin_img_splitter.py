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


if __name__ == '__main__':
    # 加载图像
    img = Image.open("../example.jpg")

    # 初始化 PatchExtractor
    # patch_size=300, stride=80, noIP=False（使用图像金字塔）
    extractor = PatchExtractor(img, patch_size=300, stride=100, noIP=False)

    # 提取 patches
    patches = extractor.extract_patches()

    # 查看提取的 patches 数量
    print(f"Total patches extracted: {len(patches)}")

    # 显示第一个 patch
    patches[0].show()