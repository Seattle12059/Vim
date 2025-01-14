import torch
from torchvision import transforms
from PIL import Image

# 从你的代码中导入 VisionMamba 和相关函数
from models_mamba import VisionMamba, vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

# 加载一张测试图像
image_path = 'path_to_your_image.jpg'  # 替换为你的图像路径
image = Image.open(image_path).convert('RGB')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])
image_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

# 初始化模型
model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False)

# 设置为评估模式
model.eval()

# 前向传播
with torch.no_grad():
    output = model(image_tensor)

# 输出结果
print("Output logits:", output)
print("Predicted class:", torch.argmax(output, dim=1).item())