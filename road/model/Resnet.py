import torch.nn as nn
from torchvision import models
class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()

        self.model = list(models.resnet50(pretrained = True).children())[:-3]
        self.features = nn.Sequential(*self.model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_extractor_part2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=self.feature_extractor_part2(x)
        return x


if __name__ == '__main__':
    import torch

    # 初始化模型
    model = Resnet()

    # 创建一个随机输入张量（模拟一批图像）
    # 假设输入图像的形状为 (batch_size, channels, height, width)
    # 这里假设输入图像大小为 224x224，3 通道
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    # 前向传播
    output = model(input_tensor)

    # 打印输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

