# test_unified.py
# 统一低光饱和去模糊模型推理脚本（支持原始模型和改进模型）

import os
import argparse
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import lpips
from piq import ssim, psnr
import importlib.util



# ----------------------------
# 动态加载模型
# ----------------------------
def load_model(model_type, model_path, device):
    """根据模型类型动态加载模型"""
    if model_type == 'original':
        module_name = 'deblur_unet_model'
    elif model_type == 'enhanced':
        module_name = 'new_unet_model'
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    # 动态导入模块
    if module_name == 'deblur_unet_model':
        from deblur_unet_model import LowLightDeblurNet
    elif module_name == 'new_unet_model':
        from new_unet_model import LowLightDeblurNet
    
    model = LowLightDeblurNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

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

        print(f"[Dataset] Found {len(self.input_paths)} input images")
        print(f"[Dataset] Found {len(self.target_paths)} target images")

        if not self.input_paths or not self.target_paths:
            raise RuntimeError("No image files found")
        if len(self.input_paths) != len(self.target_paths):
            raise RuntimeError(f"Input/target mismatch: {len(self.input_paths)} vs {len(self.target_paths)}")

        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        in_path = self.input_paths[idx]
        tgt_path = self.target_paths[idx]
        
        input_img = Image.open(in_path).convert('RGB')
        target_img = Image.open(tgt_path).convert('RGB')
        
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            
        return input_img, target_img, os.path.basename(in_path)

# ----------------------------
# 图像处理函数
# ----------------------------
def process_image(model, device, image_path, img_size=256, save_comparison=True, output_dir='output'):
    """处理单张图像并保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    input_img = Image.open(image_path).convert('RGB')
    original_size = input_img.size
    
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor).clamp(0, 1)
    
    to_pil = transforms.ToPILImage()
    output_pil = to_pil(output_tensor.squeeze(0).cpu())
    
    if output_pil.size != original_size:
        output_pil = output_pil.resize(original_size, Image.BILINEAR)
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"enhanced_{filename}")
    output_pil.save(output_path)
    print(f"Saved enhanced image: {output_path}")
    
    if save_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(input_img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        axes[1].imshow(output_pil)
        axes[1].set_title("Enhanced Result")
        axes[1].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, f"comparison_{filename}")
        plt.savefig(comparison_path, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison: {comparison_path}")
    
    return output_path

# ----------------------------
# 数据集评估函数
# ----------------------------
def evaluate_on_dataset(model, device, data_dir, img_size=256, output_dir='evaluation_results', 
                        max_samples=None, save_samples=True):
    """在整个数据集上评估模型性能"""
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
    
    print(f"Evaluating on {len(dataset)} images...")
    
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
            
            # 计算指标
            batch_psnr = psnr(outputs, targets, data_range=1.0).item()
            batch_ssim = ssim(outputs, targets, data_range=1.0).item()
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            
            # LPIPS (映射到 [-1,1])
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
            
            # 保存样本结果
            if save_samples and (idx < 10 or idx % 50 == 0):
                to_pil = transforms.ToPILImage()
                input_img = to_pil(inputs.squeeze(0).cpu())
                output_img = to_pil(outputs.squeeze(0).cpu())
                target_img = to_pil(targets.squeeze(0).cpu())
                
                # 创建三列对比图
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(input_img)
                axes[0].set_title(f"Input\n{filename[0]}")
                axes[0].axis('off')
                
                axes[1].imshow(output_img)
                axes[1].set_title(f"Enhanced\nPSNR: {batch_psnr:.2f} dB")
                axes[1].axis('off')
                
                axes[2].imshow(target_img)
                axes[2].set_title("Target")
                axes[2].axis('off')
                
                plt.tight_layout()
                sample_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
                plt.savefig(sample_path, bbox_inches='tight')
                plt.close()
    
    # 计算平均指标
    num_samples = len(metrics) if not max_samples else min(max_samples, len(dataset))
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples
    avg_time = sum(processing_times) / len(processing_times)
    fps = 1.0 / avg_time
    
    # 打印结果
    print("\n" + "="*50)
    print(f"Evaluation completed ({num_samples} images)")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Processing time: {avg_time:.4f} sec/img")
    print(f"Speed: {fps:.2f} FPS")
    print("="*50)
    
    # 保存指标结果
    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, 'w') as f:
        f.write("filename,psnr,ssim,lpips\n")
        for m in metrics:
            f.write(f"{m['filename']},{m['psnr']},{m['ssim']},{m['lpips']}\n")
    print(f"Metrics saved to: {metrics_path}")
    
    # 绘制指标分布图
    plot_metrics_distribution(metrics, output_dir)
    
    return avg_psnr, avg_ssim, metrics

def plot_metrics_distribution(metrics, output_dir):
    """绘制指标分布图"""
    psnr_values = [m['psnr'] for m in metrics]
    ssim_values = [m['ssim'] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PSNR 分布
    ax1.hist(psnr_values, bins=30, alpha=0.7, color='skyblue')
    ax1.axvline(x=np.mean(psnr_values), color='red', linestyle='dashed', linewidth=2)
    ax1.set_title('PSNR Distribution')
    ax1.set_xlabel('PSNR (dB)')
    ax1.set_ylabel('Count')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # SSIM 分布
    ax2.hist(ssim_values, bins=30, alpha=0.7, color='lightgreen')
    ax2.axvline(x=np.mean(ssim_values), color='red', linestyle='dashed', linewidth=2)
    ax2.set_title('SSIM Distribution')
    ax2.set_xlabel('SSIM')
    ax2.set_ylabel('Count')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "metrics_distribution.png")
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {dist_path}")

# ----------------------------
# 主函数
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='统一低光去模糊模型测试脚本')
    parser.add_argument('--model_path', required=True, help='训练好的模型权重路径')
    parser.add_argument('--model_type', required=True, choices=['original', 'enhanced'], 
                        help='模型类型: original (原始) 或 enhanced (改进)')
    parser.add_argument('--input_path', default=None, help='输入图像或目录路径')
    parser.add_argument('--data_dir', default=None, help='数据集根目录 (用于评估)')
    parser.add_argument('--output_dir', default='output', help='输出目录')
    parser.add_argument('--img_size', type=int, default=256, help='输入图像尺寸')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--save_comparison', action='store_true', 
                        help='保存输入/输出对比图')
    parser.add_argument('--evaluate', action='store_true', 
                        help='在数据集上全面评估模型')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='评估时的最大样本数')
    parser.add_argument('--save_samples', action='store_true', 
                        help='评估时保存样本结果')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    print(f"Model type: {args.model_type}")
    
    # 加载模型
    model = load_model(args.model_type, args.model_path, args.device)
    print("Model loaded successfully")
    
    # 运行模式
    if args.evaluate:
        if not args.data_dir:
            raise ValueError("Evaluation requires --data_dir")
            
        evaluate_on_dataset(
            model, args.device, args.data_dir, 
            img_size=args.img_size,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            save_samples=args.save_samples
        )
    elif args.input_path:
        if os.path.isdir(args.input_path):
            exts = ('*.png', '*.jpg', '*.jpeg')
            image_paths = []
            for ext in exts:
                image_paths += glob(os.path.join(args.input_path, '**', ext), recursive=True)
            
            print(f"Processing {len(image_paths)} images")
            
            for image_path in image_paths:
                process_image(
                    model, args.device, image_path, 
                    img_size=args.img_size,
                    save_comparison=args.save_comparison,
                    output_dir=args.output_dir
                )
        else:
            process_image(
                model, args.device, args.input_path, 
                img_size=args.img_size,
                save_comparison=args.save_comparison,
                output_dir=args.output_dir
            )
        
        print("Processing completed!")
    else:
        print("Error: Specify --evaluate or --input_path")

if __name__ == '__main__':
    main()