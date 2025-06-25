# train.py
# 增强版训练脚本：针对低光饱和去模糊任务的专用模型架构

import os
import argparse
import math
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
from piq import ssim, psnr  # 安装: pip install piq
import matplotlib as mpl
from deblur_unet_model import LowLightDeblurNet


# ----------------------------
# 自定义 Dataset 类
# ----------------------------
class LowLightDeblurDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        exts = ('*.png', '*.jpg', '*.jpeg')
        self.input_paths = []
        self.target_paths = []
        for ext in exts:
            self.input_paths += glob(os.path.join(root_dir, 'low_blur_noise', '**', ext), recursive=True)
            self.target_paths += glob(os.path.join(root_dir, 'high_sharp_scaled', '**', ext), recursive=True)

        self.input_paths.sort()
        self.target_paths.sort()

        print(f"[Dataset] Found {len(self.input_paths)} input images under 'low_blur_noise' recursively.")
        print(f"[Dataset] Found {len(self.target_paths)} target images under 'high_sharp_scaled' recursively.")

        if not self.input_paths or not self.target_paths:
            raise RuntimeError(f"数据目录中没有图像文件，请检查路径是否正确。")
        if len(self.input_paths) != len(self.target_paths):
            raise RuntimeError(f"输入图像数量与目标图像数量不一致：{len(self.input_paths)} vs {len(self.target_paths)}")

        self.transform = transform
        self.first_sample = None  # 保存第一个样本用于可视化

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        in_path = self.input_paths[idx]
        tgt_path = self.target_paths[idx]
        
        # 保存第一个样本用于后续可视化
        if idx == 0 and self.first_sample is None:
            print(f"[Dataset] First sample: input='{in_path}', target='{tgt_path}'")
            self.first_sample = (in_path, tgt_path)
        
        input_img = Image.open(in_path).convert('RGB')
        target_img = Image.open(tgt_path).convert('RGB')
        
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            
        return input_img, target_img

# ----------------------------
# 混合损失函数（简化版）
# ----------------------------
class HybridLoss(nn.Module):
    """
    混合损失函数，结合多种损失类型：
    1. L1 像素损失（弱光增强与亮度校正）: 60%
    2. Perceptual Loss（VGG 特征域）: 30%
    3. SSIM 损失: 10%
    """
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, device='cuda'):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha  # L1损失权重
        self.beta = beta    # 感知损失权重
        self.gamma = gamma  # SSIM损失权重
        
        # 使用预训练的VGG16网络提取特征用于感知损失
        self.vgg = None
        if beta > 0:
            self.vgg = self._build_vgg(device)
            for param in self.vgg.parameters():
                param.requires_grad = False
    
    def _build_vgg(self, device):
        """构建VGG16特征提取器"""
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        vgg = vgg.features[:16]  # 取前16层
        vgg = vgg.to(device).eval()
        return vgg
    
    def perceptual_loss(self, pred, target):
        """计算感知损失"""
        if self.vgg is None:
            return 0
        
        # 归一化到VGG的输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # 提取特征并计算L1损失
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        return F.l1_loss(pred_features, target_features)
    
    def ssim_loss(self, pred, target):
        """计算SSIM损失 (1 - SSIM)"""
        ssim_value = ssim(pred, target, data_range=1.0)
        return 1.0 - ssim_value
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        percep = self.perceptual_loss(pred, target) if self.beta > 0 else 0
        ssim_loss_val = self.ssim_loss(pred, target) if self.gamma > 0 else 0
        
        total_loss = self.alpha * l1 + self.beta * percep + self.gamma * ssim_loss_val
        
        return total_loss, {
            'l1': l1.item(), 
            'percep': percep if isinstance(percep, float) else percep.item(), 
            'ssim': ssim_loss_val.item()
        }

# ----------------------------
# 训练和验证主逻辑
# ----------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] 使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    # 创建数据集
    full_dataset = LowLightDeblurDataset(args.data_dir, transform)
    
    # 划分训练集和验证集 (验证集比例0.05)
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )
    
    print(f"[Train] 训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == 'cuda'
    )
    
    # 初始化模型
    model = LowLightDeblurNet().to(device)
    
    # 混合损失函数
    criterion = HybridLoss(
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma,
        device=device
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'psnr': [],
        'ssim': []
    }
    
    # 获取第一个样本用于可视化
    first_sample = full_dataset.first_sample
    if first_sample:
        input_path, target_path = first_sample
        input_img = transform(Image.open(input_path).convert('RGB')).unsqueeze(0).to(device)
        target_img = transform(Image.open(target_path).convert('RGB')).unsqueeze(0).to(device)
        print(f"[Visual] 使用样本进行可视化: input='{input_path}', target='{target_path}'")
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        train_metrics = {'l1': 0, 'percep': 0, 'ssim': 0}
        
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}/{args.epochs}")
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss, metrics = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失和指标
            train_loss += loss.item()
            for k in metrics:
                train_metrics[k] += metrics[k]
            
            # 更新进度条
            if (i + 1) % args.log_interval == 0:
                pbar.set_postfix(loss=loss.item())
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # 验证阶段
        val_loss, val_metrics, avg_psnr, avg_ssim = validate(model, criterion, val_loader, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['psnr'].append(avg_psnr)
        history['ssim'].append(avg_ssim)
        
        # 打印epoch总结
        print(f"\n[Epoch {epoch}] 训练完成")
        print(f"  训练损失: {avg_train_loss:.6f} (L1: {train_metrics['l1']:.6f}, Percep: {train_metrics['percep']:.6f}, SSIM: {train_metrics['ssim']:.6f})")
        print(f"  验证损失: {val_loss:.6f} (L1: {val_metrics['l1']:.6f}, Percep: {val_metrics['percep']:.6f}, SSIM: {val_metrics['ssim']:.6f})")
        print(f"  PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")
        print(f"  学习率: {current_lr:.2e}")
        
        # 保存模型检查点
        checkpoint_path = os.path.join(args.ckpt_dir, f"deblurnet_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Checkpoint] 模型已保存至: {checkpoint_path}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.ckpt_dir, "deblurnet_best.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"[Best Model] 新的最佳模型已保存至: {best_checkpoint_path}")
        
        # 使用第一个样本生成可视化结果
        if first_sample:
            model.eval()
            with torch.no_grad():
                output_img = model(input_img).clamp(0, 1)
            
            # 保存可视化结果
            save_visual_comparison(
                input_img.cpu(), 
                output_img.cpu(), 
                target_img.cpu(), 
                epoch, 
                avg_psnr, 
                avg_ssim,
                args.result_dir
            )
            print(f"[Visual] 可视化结果已保存至 {args.result_dir}/epoch_{epoch}.png")
        
        # 绘制并保存训练历史
        plot_training_history(history, os.path.join(args.result_dir, 'training_history.png'))
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(args.ckpt_dir, "deblurnet_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"[Final Model] 最终模型已保存至: {final_checkpoint_path}")

# ----------------------------
# 验证函数
# ----------------------------
def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    total_psnr = 0
    total_ssim = 0
    metrics = {'l1': 0, 'percep': 0, 'ssim': 0}
    count = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="验证中"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 添加尺寸检查
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=True)
            # 计算损失
            loss, batch_metrics = criterion(outputs, targets)
            val_loss += loss.item()
            
            # 累加指标
            for k in batch_metrics:
                metrics[k] += batch_metrics[k]
            
            # 计算PSNR和SSIM
            total_psnr += psnr(outputs, targets, data_range=1.0).item()
            total_ssim += ssim(outputs, targets, data_range=1.0).item()
            count += inputs.size(0)
    
    # 计算平均值
    avg_val_loss = val_loss / len(val_loader)
    for k in metrics:
        metrics[k] /= len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    
    return avg_val_loss, metrics, avg_psnr, avg_ssim

# ----------------------------
# 保存可视化比较结果
# ----------------------------
def save_visual_comparison(input_img, output_img, target_img, epoch, psnr_val, ssim_val, save_dir):
    # 转换为PIL图像
    to_pil = transforms.ToPILImage()
    input_pil = to_pil(input_img.squeeze(0))
    output_pil = to_pil(output_img.squeeze(0))
    target_pil = to_pil(target_img.squeeze(0))
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 输入图像
    axes[0].imshow(input_pil)
    axes[0].set_title("Low-Light Blurred Input")
    axes[0].axis('off')
    
    # 输出图像
    axes[1].imshow(output_pil)
    axes[1].set_title(f"Predicted Output\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    axes[1].axis('off')
    
    # 目标图像
    axes[2].imshow(target_pil)
    axes[2].set_title("Sharp Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"), bbox_inches='tight')
    plt.close()

# ----------------------------
# 绘制训练历史
# ----------------------------
def plot_training_history(history, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Training Loss')
    ax1.plot(epochs, history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制指标曲线
    ax2.plot(epochs, history['psnr'], label='PSNR', color='green')
    ax2.set_title('PSNR (dB)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('dB')
    ax2.grid(True)
    
    ax3.plot(epochs, history['ssim'], label='SSIM', color='purple')
    ax3.set_title('SSIM')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ----------------------------
# 命令行参数解析入口
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="低光饱和去模糊专用模型训练")
    parser.add_argument('--data_dir', default='dataset', help='包含 low_blur_noise 和 high_sharp_scaled 的根目录')
    parser.add_argument('--ckpt_dir', default='checkpoint', help='模型 checkpoint 保存目录')
    parser.add_argument('--result_dir', default='results', help='结果和可视化保存目录')
    parser.add_argument('--img_size', type=int, default=256, help='统一图像大小')
    parser.add_argument('--batch_size', type=int, default=8, help='训练 batch 大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--log_interval', type=int, default=10, help='每隔多少 batch 显示一次 loss')
    
    # 混合损失参数
    parser.add_argument('--alpha', type=float, default=0.6, help='L1损失权重')
    parser.add_argument('--beta', type=float, default=0.3, help='感知损失权重')
    parser.add_argument('--gamma', type=float, default=0.1, help='SSIM损失权重')
    
    args = parser.parse_args()

    print(f"[Args] 参数设置: {args}")
    train(args)