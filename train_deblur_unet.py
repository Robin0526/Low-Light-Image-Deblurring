# train.py
# Enhanced training script for low-light deblurring tasks

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
from piq import ssim, psnr
import time
import lpips
from deblur_unet_model import LowLightDeblurNet

# ----------------------------
# Custom Dataset Class
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

        print(f"[Dataset] Found {len(self.input_paths)} input images")
        print(f"[Dataset] Found {len(self.target_paths)} target images")

        if not self.input_paths or not self.target_paths:
            raise RuntimeError("No image files found in data directory")
        if len(self.input_paths) != len(self.target_paths):
            raise RuntimeError(f"Input/target count mismatch: {len(self.input_paths)} vs {len(self.target_paths)}")

        self.transform = transform
        self.first_sample = None

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        in_path = self.input_paths[idx]
        tgt_path = self.target_paths[idx]
        
        if idx == 0 and self.first_sample is None:
            print(f"[Dataset] First sample: input='{in_path}', target='{tgt_path}'")
            self.first_sample = (in_path, tgt_path)
        
        input_img = Image.open(in_path).convert('RGB')
        target_img = Image.open(tgt_path).convert('RGB')
        
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            
        return input_img, target_img, os.path.basename(in_path)

# ----------------------------
# Hybrid Loss Function
# ----------------------------
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, device='cuda'):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.vgg = None
        if beta > 0:
            self.vgg = self._build_vgg(device)
            for param in self.vgg.parameters():
                param.requires_grad = False
    
    def _build_vgg(self, device):
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        vgg = vgg.features[:16]
        vgg = vgg.to(device).eval()
        return vgg
    
    def perceptual_loss(self, pred, target):
        if self.vgg is None:
            return 0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        return F.l1_loss(pred_features, target_features)
    
    def ssim_loss(self, pred, target):
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
# Training and Validation
# ----------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    full_dataset = LowLightDeblurDataset(args.data_dir, transform)
    
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42))
    
    print(f"[Train] Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == 'cuda'
    )
    
    model = LowLightDeblurNet().to(device)
    
    criterion = HybridLoss(
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma,
        device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_psnr': [],
        'test_ssim': [],
        'lr': [],
        'psnr': [],
        'ssim': []
    }
    
    first_sample = full_dataset.first_sample
    if first_sample:
        input_path, target_path = first_sample
        input_img = transform(Image.open(input_path).convert('RGB')).unsqueeze(0).to(device)
        target_img = transform(Image.open(target_path).convert('RGB')).unsqueeze(0).to(device)
        print(f"[Visual] Visualization sample: input='{input_path}', target='{target_path}'")
    
    test_dataset = LowLightDeblurDataset(args.test_dir, transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == 'cuda'
    )
    print(f"[Test] Test set size: {len(test_dataset)}")
    
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        train_metrics = {'l1': 0, 'percep': 0, 'ssim': 0}
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}/{args.epochs}")
        for i, (inputs, targets, _) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss, metrics = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            for k in metrics:
                train_metrics[k] += metrics[k]
            
            if (i + 1) % args.log_interval == 0:
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        val_loss, val_metrics, avg_psnr, avg_ssim = validate(model, criterion, val_loader, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['psnr'].append(avg_psnr)
        history['ssim'].append(avg_ssim)
        
        if epoch % args.test_interval == 0:
            test_psnr, test_ssim = evaluate_on_test_set(model, test_loader, device)
            history['test_psnr'].append(test_psnr)
            history['test_ssim'].append(test_ssim)
            print(f"[Test] Test PSNR: {test_psnr:.4f} dB, SSIM: {test_ssim:.4f}")
        
        print(f"\n[Epoch {epoch}] Training complete")
        print(f"  Train loss: {avg_train_loss:.6f} (L1: {train_metrics['l1']:.6f}, Percep: {train_metrics['percep']:.6f}, SSIM: {train_metrics['ssim']:.6f})")
        print(f"  Val loss: {val_loss:.6f} (L1: {val_metrics['l1']:.6f}, Percep: {val_metrics['percep']:.6f}, SSIM: {val_metrics['ssim']:.6f})")
        print(f"  Val PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")
        print(f"  Learning rate: {current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.ckpt_dir, "deblurnet_best.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"[Best Model] Saved to: {best_checkpoint_path}")
        
        if first_sample:
            model.eval()
            with torch.no_grad():
                output_img = model(input_img).clamp(0, 1)
            
            save_visual_comparison(
                input_img.cpu(), 
                output_img.cpu(), 
                target_img.cpu(), 
                epoch, 
                avg_psnr, 
                avg_ssim,
                args.result_dir
            )
            print(f"[Visual] Saved to {args.result_dir}/epoch_{epoch}.png")
        
        plot_training_history(history, os.path.join(args.result_dir, 'training_history.png'), args.test_interval)
    
    final_checkpoint_path = os.path.join(args.ckpt_dir, "deblurnet_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"[Final Model] Saved to: {final_checkpoint_path}")
    
    print("\n[Evaluation] Evaluating final model on test set...")
    avg_psnr, avg_ssim, metrics = evaluate_on_dataset(
        model, device, args.test_dir, 
        img_size=args.img_size, 
        output_dir=os.path.join(args.result_dir, 'test_results')
    )
    print(f"[Evaluation] Final model - PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")

# ----------------------------
# Validation Function
# ----------------------------
def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    total_psnr = 0
    total_ssim = 0
    metrics = {'l1': 0, 'percep': 0, 'ssim': 0}
    
    with torch.no_grad():
        for inputs, targets, _ in tqdm(val_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=True)
            
            loss, batch_metrics = criterion(outputs, targets)
            val_loss += loss.item()
            
            for k in batch_metrics:
                metrics[k] += batch_metrics[k]
            
            total_psnr += psnr(outputs, targets, data_range=1.0).item()
            total_ssim += ssim(outputs, targets, data_range=1.0).item()
    
    avg_val_loss = val_loss / len(val_loader)
    for k in metrics:
        metrics[k] /= len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    
    return avg_val_loss, metrics, avg_psnr, avg_ssim

# ----------------------------
# Test Set Evaluation
# ----------------------------
def evaluate_on_test_set(model, test_loader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=True)
            
            total_psnr += psnr(outputs, targets, data_range=1.0).item()
            total_ssim += ssim(outputs, targets, data_range=1.0).item()
    
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    
    return avg_psnr, avg_ssim

# ----------------------------
# Dataset Evaluation
# ----------------------------
def evaluate_on_dataset(model, device, data_dir, img_size=256, output_dir='evaluation_results', 
                        max_samples=None, save_samples=True):
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    dataset = LowLightDeblurDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    total_psnr = total_ssim = total_lpips = 0.0
    metrics = []
    processing_times = []
    
    print(f"[Evaluation] Evaluating on {len(dataset)} images...")
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets, filename) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_samples and idx >= max_samples:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            outputs = torch.clamp(outputs, 0.0, 1.0)
            
            batch_psnr = psnr(outputs, targets, data_range=1.0).item()
            batch_ssim = ssim(outputs, targets, data_range=1.0).item()
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            
            out_n = outputs * 2 - 1
            tgt_n = targets * 2 - 1
            b_lpips = lpips_fn(out_n, tgt_n).item()
            total_lpips += b_lpips
            
            metrics.append({
                'filename': filename[0],
                'psnr': batch_psnr,
                'ssim': batch_ssim,
                'lpips': b_lpips
            })
            
            if save_samples and (idx < 10 or idx % 50 == 0):
                torch.cuda.synchronize()
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(inputs.squeeze(0).permute(1, 2, 0).cpu().numpy())
                axes[0].set_title(f"Input\n{filename[0]}")
                axes[0].axis('off')
                
                axes[1].imshow(outputs.squeeze(0).permute(1, 2, 0).cpu().numpy())
                axes[1].set_title(f"Enhanced\nPSNR: {batch_psnr:.2f} dB")
                axes[1].axis('off')
                
                axes[2].imshow(targets.squeeze(0).permute(1, 2, 0).cpu().numpy())
                axes[2].set_title("Target")
                axes[2].axis('off')
                
                plt.tight_layout()
                sample_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
                plt.savefig(sample_path, bbox_inches='tight')
                plt.close()
    
    num_samples = len(metrics) if not max_samples else min(max_samples, len(dataset))
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples
    avg_time = sum(processing_times) / len(processing_times)
    fps = 1.0 / avg_time
    
    print("\n" + "="*50)
    print(f"Evaluation complete ({num_samples} images)")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Processing time: {avg_time:.4f} sec/img")
    print(f"Speed: {fps:.2f} FPS")
    print("="*50)
    
    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, 'w') as f:
        f.write("filename,psnr,ssim,lpips\n")
        for m in metrics:
            f.write(f"{m['filename']},{m['psnr']},{m['ssim']},{m['lpips']}\n")
    print(f"Metrics saved to: {metrics_path}")
    
    return avg_psnr, avg_ssim, metrics

# ----------------------------
# Visualization
# ----------------------------
def save_visual_comparison(input_img, output_img, target_img, epoch, psnr_val, ssim_val, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_img.squeeze(0).permute(1, 2, 0).numpy())
    axes[0].set_title("Low-light Input")
    axes[0].axis('off')
    
    axes[1].imshow(output_img.squeeze(0).permute(1, 2, 0).numpy())
    axes[1].set_title(f"Enhanced\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    axes[1].axis('off')
    
    axes[2].imshow(target_img.squeeze(0).permute(1, 2, 0).numpy())
    axes[2].set_title("Target")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"), bbox_inches='tight')
    plt.close()

# ----------------------------
# Training History Plot
# ----------------------------
def plot_training_history(history, save_path, test_interval):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    
    if history['test_psnr']:
        test_epochs = [e for i, e in enumerate(epochs) if (i+1) % test_interval == 0]
        ax1.scatter(test_epochs, 
                   [history['val_loss'][i] for i in range(len(history['val_loss'])) if (i+1) % test_interval == 0], 
                   color='red', s=50, label='Test Points')
    
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['psnr'], label='Val PSNR', color='blue')
    if history['test_psnr']:
        test_epochs = [e for i, e in enumerate(epochs) if (i+1) % test_interval == 0]
        ax2.scatter(test_epochs, history['test_psnr'], color='red', s=50, label='Test PSNR')
    ax2.set_title('PSNR (dB)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('dB')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(epochs, history['ssim'], label='Val SSIM', color='purple')
    if history['test_ssim']:
        test_epochs = [e for i, e in enumerate(epochs) if (i+1) % test_interval == 0]
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
# Main Entry Point
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Low-light Deblurring Model Training")
    parser.add_argument('--data_dir', default='dataset/train', help='Root directory for training data')
    parser.add_argument('--test_dir', default='dataset/test', help='Test dataset directory')
    parser.add_argument('--ckpt_dir', default='checkpoint', help='Checkpoint directory')
    parser.add_argument('--result_dir', default='result', help='Results directory')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--test_interval', type=int, default=5, help='Test evaluation interval')
    
    parser.add_argument('--alpha', type=float, default=0.6, help='L1 loss weight')
    parser.add_argument('--beta', type=float, default=0.3, help='Perceptual loss weight')
    parser.add_argument('--gamma', type=float, default=0.1, help='SSIM loss weight')
    
    args = parser.parse_args()

    print(f"[Args] Parameters: {args}")
    train(args)