import torch
import torchvision.models as models

# 加载预训练的 EfficientNet-B3 模型
model = models.efficientnet_b3(pretrained=True)

# 移除最后的分类层
model = torch.nn.Sequential(*list(model.children())[:-1])

# 设置模型为评估模式
model.eval()

# 示例输入 patch (假设已经预处理为 PyTorch tensor)
patch = torch.randn(1, 3, 300, 300) #  (Batch size, Channels, Height, Width)

# 提取特征
with torch.no_grad():
    features = model(patch)


# 全局平均池化 (可选)
features = torch.mean(features, dim=(2, 3))

print(features.shape) #  输出特征维度
