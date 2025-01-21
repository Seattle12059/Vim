import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


class EfficientNetPatchEmbed(nn.Module):
    """
    使用 EfficientNet-B3 作为 patch_embed 的生成器。
    将 EfficientNet-B3 提取的 1536 维特征映射到目标维度 384。
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        # 加载预训练的 EfficientNet-B3 模型
        self.efficientnet = models.efficientnet_b3(pretrained=True)

        # 移除最后的分类层，保留特征提取部分
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])

        # 添加一个线性层，将 1536 维特征映射到目标维度 384
        self.proj = nn.Linear(1536, embed_dim)

        # 归一化层
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        前向传播：
        1. 使用 EfficientNet-B3 提取特征。
        2. 将特征从 1536 维映射到 384 维。
        3. 应用归一化。
        """
        # 提取特征 (B, 1536, 1, 1)
        with torch.no_grad():
            features = self.efficientnet(x)

        # 全局平均池化 (B, 1536, 1, 1) -> (B, 1536)
        features = torch.mean(features, dim=(2, 3))

        # 映射到目标维度 (B, 1536) -> (B, 384)
        features = self.proj(features)

        # 归一化
        features = self.norm(features)

        # 调整形状为 (B, 1, 384)，模拟 patch_embed 的输出
        features = features.unsqueeze(1)
        return features


class VisionMambaWithEfficientNet(nn.Module):
    """
    修改后的 VisionMamba 模型，使用 EfficientNet-B3 作为 patch_embed 的生成器。
    """

    def __init__(self, **kwargs):
        super().__init__()

        # 使用 EfficientNetPatchEmbed 替换原始的 PatchEmbed
        self.patch_embed = EfficientNetPatchEmbed(embed_dim=384)

        # 加载预训练的 vim_small_patch16_224 模型
        self.vim_model = vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
            pretrained=True)

        # 冻结 vim_small_patch16_224 的参数（可选）
        for param in self.vim_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        前向传播：
        1. 使用 EfficientNet-B3 提取特征并映射到 384 维。
        2. 将特征输入到 vim_small_patch16_224 模型中。
        """
        # 提取特征 (B, 1, 384)
        patch_embed = self.patch_embed(x)

        # 输入到 vim_small_patch16_224 模型中
        output = self.vim_model.forward_features(patch_embed)
        return output


@register_model
def vim_small_patch16_224_new(pretrained=False, **kwargs):
    """
    注册新的模型 vim_small_patch16_224_new。
    """
    model = VisionMambaWithEfficientNet(**kwargs)
    model.default_cfg = _cfg()
    return model


# 示例用法
if __name__ == "__main__":
    # 创建模型实例
    model = vim_small_patch16_224_new(pretrained=True)
    model.eval()

    # 示例输入 (B, C, H, W)
    patch = torch.randn(1, 3, 300, 300)

    # 前向传播
    with torch.no_grad():
        output = model(patch)

    print(output.shape)  # 输出特征维度