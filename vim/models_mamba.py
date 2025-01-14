# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from rope import *
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]


class PatchEmbed(nn.Module):
    """
    [Cline注释] 这个类负责将2D图像转换为Patch Embedding，是Vision Transformer架构中的关键组件之一。
    主要功能是将输入图像分割成固定大小的patch，然后通过线性投影将每个patch映射到embedding空间。

    参数:
        img_size (int or tuple): 输入图像的尺寸，默认为224。表示输入图像的高度和宽度。
        patch_size (int or tuple): 每个patch的尺寸，默认为16。表示将图像分割成16x16的小块。
        stride (int): 卷积步长，默认为16。控制patch之间的重叠程度，通常等于patch_size。
        in_chans (int): 输入通道数，默认为3（RGB图像）。对于灰度图像可以设置为1。
        embed_dim (int): 嵌入维度，默认为768。表示每个patch将被映射到的特征维度。
        norm_layer (nn.Module): 归一化层，默认为None（即不使用归一化）。可以传入LayerNorm等归一化层。
        flatten (bool): 是否展平，默认为True。如果为True，将2D patch展平为1D序列。

    [Cline注释] 示例：
        对于224x224的输入图像，patch_size=16，stride=16：
        - 将生成 (224/16) x (224/16) = 14 x 14 = 196个patch
        - 每个patch将被映射到embed_dim维的特征空间
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # 定义卷积层用于提取patch特征
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        # 定义归一化层
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        执行前向传播过程，对输入的图像进行处理。

        参数:
            x (Tensor): 输入的图像张量，形状为(B, C, H, W)。

        返回:
            Tensor: 经过前向传播处理后的张量。
        """
        B, C, H, W = x.shape
        # 确保输入图像的尺寸与模型预期的尺寸相匹配
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 对输入图像进行投影
        x = self.proj(x)
        # 如果设置了flatten标志，将图像张量展平并进行转置
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # 应用归一化
        x = self.norm(x)
        return x


class Block(nn.Module):
    """
    [Cline注释] Block类是Vision Mamba模型的基本构建块，包含以下主要组件：
    1. Mixer层：负责特征混合和转换
    2. 归一化层：用于稳定训练过程
    3. 残差连接：帮助梯度流动，防止梯度消失

    参数:
        dim (int): 输入和输出的特征维度。控制每个token的表示维度。
        mixer_cls (callable): Mixer类的构造函数。用于创建特征混合层。
        norm_cls (nn.Module): 归一化层，默认为nn.LayerNorm。可以选择LayerNorm或RMSNorm。
        fused_add_norm (bool): 是否使用融合的加法和归一化，默认为False。可以加速计算。
        residual_in_fp32 (bool): 是否在FP32中保留残差，默认为False。有助于数值稳定性。
        drop_path (float): 随机深度概率，默认为0。用于实现随机深度正则化。

    [Cline注释] 前向传播流程：
    1. 输入特征通过Mixer层进行特征转换
    2. 应用残差连接
    3. 通过归一化层进行特征归一化
    4. 返回处理后的特征和残差

    [Cline注释] 典型应用场景：
    - 作为Vision Mamba模型的基本构建块
    - 用于构建深度神经网络
    - 适用于需要长序列建模的视觉任务
    """
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # 定义Mixer层
        self.mixer = mixer_cls(dim)
        # 定义归一化层
        self.norm = norm_cls(dim)
        # 定义DropPath层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        """
        通过编码器层传递输入。

        参数:
            hidden_states (Tensor): 输入序列（必须提供）。
            residual (Optional[Tensor]): 残差张量（可选），默认为None。
            inference_params: 推理参数（可选）。

        返回:
            Tuple[Tensor, Tensor]: 处理后的隐藏状态和残差。
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        分配推理缓存。

        参数:
            batch_size (int): 批次大小。
            max_seqlen (int): 最大序列长度。
            dtype (torch.dtype): 数据类型，默认为None。
            kwargs: 其他参数。

        返回:
            Dict: 推理缓存。
        """
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    """
    创建一个Block实例。

    参数:
        d_model (int): 模型维度。
        d_state (int): 状态维度，默认为16。
        ssm_cfg (dict): SSM配置，默认为None。
        norm_epsilon (float): 归一化层的epsilon，默认为1e-5。
        drop_path (float): 随机深度概率，默认为0。
        rms_norm (bool): 是否使用RMS归一化，默认为False。
        residual_in_fp32 (bool): 是否在FP32中保留残差，默认为False。
        fused_add_norm (bool): 是否使用融合的加法和归一化，默认为False。
        layer_idx (int): 层索引，默认为None。
        device (torch.device): 设备，默认为None。
        dtype (torch.dtype): 数据类型，默认为None。
        if_bimamba (bool): 是否使用BiMamba，默认为False。
        bimamba_type (str): BiMamba类型，默认为"none"。
        if_divide_out (bool): 是否除以输出，默认为False。
        init_layer_scale (float): 初始化层缩放，默认为None。

    返回:
        Block: 创建的Block实例。
    """
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    """
    初始化权重。

    参数:
        module (nn.Module): 要初始化的模块。
        n_layer (int): 层数。
        initializer_range (float): 初始化范围，默认为0.02。
        rescale_prenorm_residual (bool): 是否重新缩放预归一化残差，默认为True。
        n_residuals_per_layer (int): 每层的残差数量，默认为1。
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    """
    初始化分割权重。

    参数:
        m (nn.Module): 要初始化的模块。
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    """
    [Cline注释] VisionMamba是基于Mamba架构的视觉Transformer模型，主要特点包括：
    1. 使用Mamba块进行特征提取
    2. 支持多种位置编码方式
    3. 灵活的架构配置
    4. 高效的序列建模能力

    参数:
        img_size (int): 输入图像的尺寸，默认为224。表示输入图像的高度和宽度。
        patch_size (int): 每个patch的尺寸，默认为16。将图像分割成16x16的小块。
        stride (int): 卷积步长，默认为16。控制patch之间的重叠程度。
        depth (int): 模型深度，默认为24。表示Mamba块的数量。
        embed_dim (int): 嵌入维度，默认为192。控制每个patch的特征维度。
        d_state (int): 状态维度，默认为16。控制Mamba块的状态空间大小。
        channels (int): 输入通道数，默认为3（RGB图像）。对于灰度图像可以设置为1。
        num_classes (int): 类别数量，默认为1000。控制分类头的输出维度。
        ssm_cfg (dict): SSM配置，默认为None。用于配置状态空间模型参数。
        drop_rate (float): dropout概率，默认为0。用于防止过拟合。
        drop_path_rate (float): 随机深度概率，默认为0.1。用于随机深度正则化。
        norm_epsilon (float): 归一化层的epsilon，默认为1e-5。用于数值稳定性。
        rms_norm (bool): 是否使用RMS归一化，默认为True。选择归一化方式。
        initializer_cfg (dict): 初始化配置，默认为None。控制模型参数初始化方式。
        fused_add_norm (bool): 是否使用融合的加法和归一化，默认为True。可以加速计算。
        residual_in_fp32 (bool): 是否在FP32中保留残差，默认为True。有助于数值稳定性。
        device (torch.device): 设备，默认为None。指定模型运行的设备。
        dtype (torch.dtype): 数据类型，默认为None。控制模型计算精度。
        ft_seq_len (int): 微调序列长度，默认为None。用于调整序列长度。
        pt_hw_seq_len (int): 预训练序列长度，默认为14。用于位置编码。
        if_bidirectional (bool): 是否双向，默认为False。控制是否使用双向Mamba。
        final_pool_type (str): 最终池化类型，默认为'none'。控制输出池化方式。
        if_abs_pos_embed (bool): 是否使用绝对位置嵌入，默认为True。控制位置编码方式。
        if_rope (bool): 是否使用旋转位置嵌入，默认为False。控制是否使用RoPE。
        if_rope_residual (bool): 是否在残差中使用旋转位置嵌入，默认为False。
        flip_img_sequences_ratio (float): 图像序列翻转比例，默认为-1。数据增强参数。
        if_cls_token (bool): 是否使用CLS标记，默认为True。控制分类token的使用。
        if_divide_out (bool): 是否除以输出，默认为True。控制输出归一化。
        init_layer_scale (float): 初始化层缩放，默认为None。控制初始化规模。
        use_double_cls_token (bool): 是否使用双CLS标记，默认为False。控制分类token数量。
        use_middle_cls_token (bool): 是否使用中间CLS标记，默认为True。控制分类token位置。
        **kwargs: 其他参数。

    [Cline注释] 主要组件：
    1. PatchEmbed: 将图像转换为patch embeddings
    2. Mamba Blocks: 多个Mamba块组成的特征提取器
    3. Classification Head: 用于最终分类的全连接层

    [Cline注释] 典型应用场景：
    - 图像分类任务
    - 视觉特征提取
    - 需要长序列建模的视觉任务
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride=16,
                 depth=24,
                 embed_dim=192,
                 d_state=16,
                 channels=3,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = True,
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_cls_token=True,
                 if_divide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # 预训练参数
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # 定义PatchEmbed层
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减规则
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # 输出头
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # 原始初始化
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba 初始化
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        """
        [Cline注释] 前向传播特征提取过程，主要步骤：
        1. 将输入图像转换为patch embeddings
        2. 添加CLS token（如果启用）
        3. 添加位置编码（如果启用）
        4. 通过Mamba块进行特征提取
        5. 返回最终特征表示

        参数:
            x (Tensor): 输入图像张量，形状为(B, C, H, W)
            inference_params: 推理参数，用于控制推理过程
            if_random_cls_token_position (bool): 是否随机放置CLS token
            if_random_token_rank (bool): 是否随机打乱token顺序

        返回:
            Tensor: 提取的特征表示
        """
        # 1. 通过PatchEmbed层将图像转换为patch embeddings
        x = self.patch_embed(x)
        B, M, _ = x.shape

        # 2. 处理CLS token
        if self.if_cls_token:
            if self.use_double_cls_token:
                # 双CLS token模式
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                # 单CLS token模式
                cls_token = self.cls_token.expand(B, -1, -1)
                if self.use_middle_cls_token:
                    # 中间位置插入CLS token
                    token_position = M // 2
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    # 随机位置插入CLS token
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                else:
                    # 开头位置插入CLS token
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        # 3. 添加绝对位置编码
        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # 4. 随机打乱token顺序（如果启用）
        if if_random_token_rank:
            shuffle_indices = torch.randperm(M)
            x = x[:, shuffle_indices, :]
            # 更新CLS token位置
            if isinstance(token_position, list):
                token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
            else:
                token_position = torch.where(shuffle_indices == token_position)[0].item()

        # 5. 随机翻转图像序列（数据增强）
        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # 6. 通过Mamba块进行特征提取
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            # 单向Mamba
            for layer in self.layers:
                # 处理序列翻转和RoPE
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # 应用RoPE
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                # 通过Mamba层
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # 双向Mamba
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                # 前向和后向Mamba
                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        # 7. 最终归一化
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # 8. 返回特征表示
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                return hidden_states[:, token_position, :]

        # 根据池化类型返回结果
        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        """
        [Cline注释] 完整的前向传播过程，包含以下步骤：
        1. 通过forward_features方法提取特征
        2. 根据return_features参数决定是否返回中间特征
        3. 通过分类头得到最终输出
        4. 根据池化类型处理输出

        参数:
            x (Tensor): 输入图像张量，形状为(B, C, H, W)
            return_features (bool): 是否返回中间特征，默认为False
            inference_params: 推理参数，用于控制推理过程
            if_random_cls_token_position (bool): 是否随机放置CLS token
            if_random_token_rank (bool): 是否随机打乱token顺序

        返回:
            Tensor: 模型输出，形状为(B, num_classes)或(B, embed_dim)
        """
        # 1. 通过forward_features方法提取特征
        x = self.forward_features(x, inference_params, 
                                if_random_cls_token_position=if_random_cls_token_position,
                                if_random_token_rank=if_random_token_rank)
        
        # 2. 如果只需要特征表示，直接返回
        if return_features:
            return x
            
        # 3. 通过分类头得到最终输出
        x = self.head(x)
        
        # 4. 根据池化类型处理输出
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
            
        return x


@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    """
    [Cline注释] Vision Mamba Tiny模型配置
    参数:
        patch_size=16: patch大小为16x16
        embed_dim=192: 嵌入维度为192
        depth=24: 24层Mamba块
        rms_norm=True: 使用RMS归一化
        residual_in_fp32=True: 在FP32中保留残差
        fused_add_norm=True: 使用融合的加法和归一化
        final_pool_type='mean': 使用均值池化
        if_abs_pos_embed=True: 使用绝对位置编码
        if_rope=False: 不使用旋转位置编码
        bimamba_type="v2": 使用双向Mamba V2
        if_cls_token=True: 使用CLS token
        if_divide_out=True: 输出除以2
        use_middle_cls_token=True: 在中间位置插入CLS token

    [Cline注释] 使用方法：
    1. 导入模型：
       from vim.models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    2. 创建模型实例：
       model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=True)
    3. 准备输入数据：
       input_tensor = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)
    4. 前向传播：
       output = model(input_tensor)
    5. 训练/推理：
       # 训练时使用交叉熵损失
       criterion = nn.CrossEntropyLoss()
       loss = criterion(output, target)
       loss.backward()
       # 推理时使用argmax获取预测类别
       pred = output.argmax(dim=1)
    """
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    """
    [Cline注释] Vision Mamba Tiny模型配置，使用stride=8
    与vim_tiny_patch16_224相比，主要区别：
        stride=8: 使用步长为8的卷积，产生更多patch
    """
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    """
    [Cline注释] Vision Mamba Small模型配置
    与Tiny版本相比，主要区别：
        embed_dim=384: 嵌入维度增加到384
    """
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    """
    [Cline注释] Vision Mamba Small模型配置，使用stride=8
    与vim_small_patch16_224相比，主要区别：
        stride=8: 使用步长为8的卷积，产生更多patch
    """
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
    
@register_model
def vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(pretrained=False, **kwargs):
    """
    [Cline注释] Vision Mamba Base模型配置
    与Small版本相比，主要区别：
        embed_dim=768: 嵌入维度增加到768
        d_state=16: 状态维度设置为16
    """
    model = VisionMamba(
        patch_size=16, embed_dim=768, d_state=16, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
