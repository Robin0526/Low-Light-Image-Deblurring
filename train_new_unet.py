# train_new_unet.py
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
from new_unet_model import LowLightDeblurNet  # 导入改进的模型

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
            raise RuntimeError(f"No image files found in the data directory. Please check the path.")
        if len(self.input_paths) != len(self.target_paths):
            raise RuntimeError(f"Input and target image counts do not match: {len(self.input_paths)} vs {len(self.target_paths)}")

        self.transform = transform
        self.first_sample = None  # Save the first sample for visualization

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        in_path = self.input_paths[idx]
        tgt_path = self.target_paths[idx]
        
        # Save the first sample for later visualization
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
    Hybrid loss function combining multiple loss types:
    1. L1 pixel loss (for low-light enhancement and brightness correction): 60%
    2. Perceptual Loss (VGG feature domain): 30%
    3. SSIM loss: 10%
    """
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, device='cuda'):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta    # perceptual loss weight
        self.gamma = gamma  # SSIM loss weight
        
        # Pre-trained VGG16 for perceptual loss
        self.vgg = None
        if beta > 0:
            self.vgg = self._build_vgg(device)
            for param in self.vgg.parameters():
                param.requires_grad = False
    
    def _build_vgg(self, device):
        """Build VGG16 feature extractor"""
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        vgg = vgg.features[:16]  # Take the first 16 layers
        vgg = vgg.to(device).eval()
        return vgg
    
    def perceptual_loss(self, pred, target):
        """Compute perceptual loss"""
        if self.vgg is None:
            return 0
        
        # Normalize to VGG input range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # Extract features and compute L1 loss
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        return F.l1_loss(pred_features, target_features)
    
    def ssim_loss(self, pred, target):
        """Compute SSIM loss (1 - SSIM)"""
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
    print(f"[Train] Using device: {device}")
    
    # Create output directories
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    full_dataset = LowLightDeblurDataset(args.data_dir, transform)
    
    # Split into train and validation sets (validation ratio 0.05)
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Fixed random seed
    )
    
    print(f"[Train] Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == 'cuda'
    )
    
    # Initialize model (using the improved model)
    model = LowLightDeblurNet().to(device)
    
    # Hybrid loss function
    criterion = HybridLoss(
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma,
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_psnr': [],   # 新增：测试集PSNR
        'test_ssim': [],   # 新增：测试集SSIM
        'lr': [],
        'psnr': [],
        'ssim': []
    }
    
    # Get the first sample for visualization
    first_sample = full_dataset.first_sample
    if first_sample:
        input_path, target_path = first_sample
        input_img = transform(Image.open(input_path).convert('RGB')).unsqueeze(0).to(device)
        target_img = transform(Image.open(target_path).convert('RGB')).unsqueeze(0).to(device)
        print(f"[Visual] Using sample for visualization: input='{input_path}', target='{target_path}'")
    
    # 创建测试集加载器（新增）
    test_dataset = LowLightDeblurDataset(args.test_dir, transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == 'cuda'
    )
    print(f"[Test] Test set size: {len(test_dataset)}")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0
        train_metrics = {'l1': 0, 'percep': 0, 'ssim': 0}
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}/{args.epochs}")
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss, metrics = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss and metrics
            train_loss += loss.item()
            for k in metrics:
                train_metrics[k] += metrics[k]
            
            # Update progress bar
            if (i + 1) % args.log_interval == 0:
                pbar.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # Validation phase
        val_loss, val_metrics, avg_psnr, avg_ssim = validate(model, criterion, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['psnr'].append(avg_psnr)
        history['ssim'].append(avg_ssim)
        
        # 周期性在测试集上评估（新增）
        if epoch % args.test_interval == 0:
            test_psnr, test_ssim = evaluate_on_test_set(model, test_loader, device)
            history['test_psnr'].append(test_psnr)
            history['test_ssim'].append(test_ssim)
            print(f"[Test] Test PSNR: {test_psnr:.4f} dB, SSIM: {test_ssim:.4f}")
        
        # Print epoch summary
        print(f"\n[Epoch {epoch}] Training completed")
        print(f"  Training loss: {avg_train_loss:.6f} (L1: {train_metrics['l1']:.6f}, Percep: {train_metrics['percep']:.6f}, SSIM: {train_metrics['ssim']:.6f})")
        print(f"  Validation loss: {val_loss:.6f} (L1: {val_metrics['l1']:.6f}, Percep: {val_metrics['percep']:.6f}, SSIM: {val_metrics['ssim']:.6f})")
        print(f"  PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")
        print(f"  Learning rate: {current_lr:.2e}")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(args.ckpt_dir, f"deblurnet_fac_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Checkpoint] Model saved to: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.ckpt_dir, "deblurnet_fac_best.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"[Best Model] New best model saved to: {best_checkpoint_path}")
        
        # Generate visualization using the first sample
        if first_sample:
            model.eval()
            with torch.no_grad():
                output_img = model(input_img).clamp(0, 1)
            
            # Save visualization
            save_visual_comparison(
                input_img.cpu(), 
                output_img.cpu(), 
                target_img.cpu(), 
                epoch, 
                avg_psnr, 
                avg_ssim,
                args.result_dir
            )
            print(f"[Visual] Visualization saved to {args.result_dir}/epoch_{epoch}.png")
        
        # Plot and save training history
        plot_training_history(history, os.path.join(args.result_dir, 'training_history.png'))
    
    # Save final model
    final_checkpoint_path = os.path.join(args.ckpt_dir, "deblurnet_fac_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"[Final Model] Final model saved to: {final_checkpoint_path}")

# ----------------------------
# Validation function
# ----------------------------
def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    total_psnr = 0
    total_ssim = 0
    metrics = {'l1': 0, 'percep': 0, 'ssim': 0}
    count = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Check and adjust size if necessary
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=True)
            # Compute loss
            loss, batch_metrics = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Accumulate metrics
            for k in batch_metrics:
                metrics[k] += batch_metrics[k]
            
            # Compute PSNR and SSIM
            total_psnr += psnr(outputs, targets, data_range=1.0).item()
            total_ssim += ssim(outputs, targets, data_range=1.0).item()
            count += inputs.size(0)
    
    # Calculate averages
    avg_val_loss = val_loss / len(val_loader)
    for k in metrics:
        metrics[k] /= len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    
    return avg_val_loss, metrics, avg_psnr, avg_ssim

# ----------------------------
# 新增：在测试集上评估函数
# ----------------------------
def evaluate_on_test_set(model, test_loader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Check and adjust size if necessary
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=True)
            
            # Compute PSNR and SSIM
            total_psnr += psnr(outputs, targets, data_range=1.0).item()
            total_ssim += ssim(outputs, targets, data_range=1.0).item()
    
    # Calculate averages
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    
    return avg_psnr, avg_ssim

# ----------------------------
# Save visual comparison
# ----------------------------
def save_visual_comparison(input_img, output_img, target_img, epoch, psnr_val, ssim_val, save_dir):
    # Convert to PIL images
    to_pil = transforms.ToPILImage()
    input_pil = to_pil(input_img.squeeze(0))
    output_pil = to_pil(output_img.squeeze(0))
    target_pil = to_pil(target_img.squeeze(0))
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axes[0].imshow(input_pil)
    axes[0].set_title("Low-Light Blurred Input")
    axes[0].axis('off')
    
    # Output image
    axes[1].imshow(output_pil)
    axes[1].set_title(f"Predicted Output\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    axes[1].axis('off')
    
    # Target image
    axes[2].imshow(target_pil)
    axes[2].set_title("Sharp Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"), bbox_inches='tight')
    plt.close()

# ----------------------------
# Plot training history
# ----------------------------
def plot_training_history(history, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Training Loss')
    ax1.plot(epochs, history['val_loss'], label='Validation Loss')
    
    # 标记测试点（新增）
    if 'test_psnr' in history and history['test_psnr']:
        test_epochs = [e for i, e in enumerate(epochs) if (i+1) % args.test_interval == 0]
        ax1.scatter(test_epochs, [history['val_loss'][i] for i in range(len(history['val_loss'])) if (i+1) % args.test_interval == 0], 
                   color='red', s=50, label='Test Evaluation')
    
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot PSNR
    ax2.plot(epochs, history['psnr'], label='Validation PSNR', color='blue')
    if 'test_psnr' in history and history['test_psnr']:
        test_epochs = [e for i, e in enumerate(epochs) if (i+1) % args.test_interval == 0]
        ax2.scatter(test_epochs, history['test_psnr'], color='red', s=50, label='Test PSNR')
    ax2.set_title('PSNR (dB)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('dB')
    ax2.legend()
    ax2.grid(True)
    
    # Plot SSIM
    ax3.plot(epochs, history['ssim'], label='Validation SSIM', color='purple')
    if 'test_ssim' in history and history['test_ssim']:
        test_epochs = [e for i, e in enumerate(epochs) if (i+1) % args.test_interval == 0]
        ax3.scatter(test_epochs, history['test_ssim'], color='red', s=50, label='Test SSIM')
    ax3.set_title('SSIM')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ----------------------------
# Command line argument parsing
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training for Low-Light Deblurring Model")
    parser.add_argument('--data_dir', default='dataset/train', help='Root directory containing low_blur_noise and high_sharp_scaled')
    parser.add_argument('--test_dir', default='dataset/test', help='Test dataset directory')  # 新增测试集参数
    parser.add_argument('--ckpt_dir', default='checkpoint', help='Directory to save model checkpoints')
    parser.add_argument('--result_dir', default='result', help='Directory to save results and visualizations')
    parser.add_argument('--img_size', type=int, default=256, help='Uniform image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval in batches')
    parser.add_argument('--test_interval', type=int, default=5, help='Evaluate on test set every N epochs')  # 新增测试间隔
    
    # Hybrid loss parameters
    parser.add_argument('--alpha', type=float, default=0.6, help='Weight for L1 loss')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for perceptual loss')
    parser.add_argument('--gamma', type=float, default=0.1, help='Weight for SSIM loss')
    
    args = parser.parse_args()

    print(f"[Args] Arguments: {args}")
    train(args)