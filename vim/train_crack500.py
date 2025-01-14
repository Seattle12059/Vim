"""
[Cline注释] 使用Vision Mamba模型训练crack500数据集的脚本

主要功能：
1. 加载crack500数据集
2. 配置Vision Mamba模型
3. 训练模型
4. 保存训练结果
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为Tensor
    transforms.Normalize(           # 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def prepare_dataset(data_dir):
    """准备crack500数据集，创建train/val目录结构"""
    import shutil
    from sklearn.model_selection import train_test_split
    
    # 创建目录结构
    os.makedirs(os.path.join(data_dir, 'train', 'diseased'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'diseased'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'normal'), exist_ok=True)
    
    # 获取所有图像路径
    diseased_imgs = [os.path.join(data_dir, 'diseased', f) 
                    for f in os.listdir(os.path.join(data_dir, 'diseased'))]
    normal_imgs = [os.path.join(data_dir, 'normal', f)
                  for f in os.listdir(os.path.join(data_dir, 'normal'))]
    
    # 划分训练集和验证集
    train_diseased, val_diseased = train_test_split(diseased_imgs, test_size=0.2, random_state=42)
    train_normal, val_normal = train_test_split(normal_imgs, test_size=0.2, random_state=42)
    
    # 复制文件到对应目录
    for img in train_diseased:
        shutil.copy(img, os.path.join(data_dir, 'train', 'diseased'))
    for img in val_diseased:
        shutil.copy(img, os.path.join(data_dir, 'val', 'diseased'))
    for img in train_normal:
        shutil.copy(img, os.path.join(data_dir, 'train', 'normal'))
    for img in val_normal:
        shutil.copy(img, os.path.join(data_dir, 'val', 'normal'))

# 数据集路径
data_dir = 'crack500'  # 请确保crack500文件夹存在并包含训练数据

# 准备数据集
prepare_dataset(data_dir)

# 加载数据集
train_dataset = ImageFolder(
    root=os.path.join(data_dir, 'train'),
    transform=transform
)

val_dataset = ImageFolder(
    root=os.path.join(data_dir, 'val'),
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# 预训练模型路径
pretrained_path = 'Vim-small-midclstok/vim_s_midclstok_ft_81p6acc.pth'  # 使用微调后的预训练模型

# 初始化模型并加载预训练权重
model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    num_classes=2  # crack500是二分类问题
)
# 加载预训练权重
model.load_state_dict(torch.load(pretrained_path), strict=False)

# 如果有GPU则使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 使用较小的学习率进行微调
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# 训练参数
num_epochs = 20  # 微调时epoch数可以适当减少
best_val_acc = 0.0

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # 训练阶段
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    
    # 打印训练信息
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {running_loss/len(train_loader):.4f}, '
          f'Val Acc: {val_acc:.2f}%')
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'New best model saved with val acc: {val_acc:.2f}%')

print('Training complete')
print(f'Best validation accuracy: {best_val_acc:.2f}%')
