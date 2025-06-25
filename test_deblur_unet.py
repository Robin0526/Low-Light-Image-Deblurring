# test.py
# 低光饱和去模糊模型推理脚本

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
import matplotlib as mpl
from piq import ssim, psnr  # 安装: pip install piq
import time
from tqdm import tqdm
from deblur_unet_model import LowLightDeblurNet

# ----------------------------
# 自定义 Dataset 类 (与训练代码相同)
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
            
        return input_img, target_img, os.path.basename(in_path)



# ----------------------------
# 图像处理函数
# ----------------------------
def process_image(model, device, image_path, img_size=256, save_comparison=True, output_dir='output'):
    """处理单张图像并保存结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # 读取图像
    input_img = Image.open(image_path).convert('RGB')
    original_size = input_img.size
    
    # 应用预处理
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor).clamp(0, 1)
    
    # 后处理
    to_pil = transforms.ToPILImage()
    output_pil = to_pil(output_tensor.squeeze(0).cpu())
    
    # 如果需要恢复原始尺寸
    if output_pil.size != original_size:
        output_pil = output_pil.resize(original_size, Image.BILINEAR)
    
    # 保存结果
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"enhanced_{filename}")
    output_pil.save(output_path)
    print(f"已保存增强图像: {output_path}")
    
    # 保存对比图
    if save_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 输入图像
        axes[0].imshow(input_img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # 输出图像
        axes[1].imshow(output_pil)
        axes[1].set_title("Enhanced Result")
        axes[1].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, f"comparison_{filename}")
        plt.savefig(comparison_path, bbox_inches='tight')
        plt.close()
        print(f"已保存对比图: {comparison_path}")
    
    return output_path

# ----------------------------
# 在整个数据集上评估模型性能
# ----------------------------
def evaluate_on_dataset(model, device, data_dir, img_size=256, output_dir='evaluation_results', 
                        max_samples=None, save_samples=True):
    """在整个数据集上评估模型性能"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # 创建数据集
    dataset = LowLightDeblurDataset(data_dir, transform=transform)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 初始化指标
    total_psnr = 0.0
    total_ssim = 0.0
    metrics = []
    processing_times = []
    
    print(f"开始在数据集上评估模型，共 {len(dataset)} 张图像...")
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets, filename) in enumerate(tqdm(dataloader, desc="评估中")):
            if max_samples and idx >= max_samples:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 测量处理时间
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            # 计算指标
            batch_psnr = psnr(outputs, targets, data_range=1.0).item()
            batch_ssim = ssim(outputs, targets, data_range=1.0).item()
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            
            # 记录每张图像的指标
            metrics.append({
                'filename': filename[0],
                'psnr': batch_psnr,
                'ssim': batch_ssim
            })
            
            # 保存样本结果
            if save_samples and (idx < 10 or idx % 50 == 0):
                # 转换为PIL图像
                to_pil = transforms.ToPILImage()
                input_img = to_pil(inputs.squeeze(0).cpu())
                output_img = to_pil(outputs.squeeze(0).cpu())
                target_img = to_pil(targets.squeeze(0).cpu())
                
                # 创建对比图
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 输入图像
                axes[0].imshow(input_img)
                axes[0].set_title(f"Input Image\n{filename[0]}")
                axes[0].axis('off')
                
                # 输出图像
                axes[1].imshow(output_img)
                axes[1].set_title(f"Enhanced Result\nPSNR: {batch_psnr:.2f} dB\nSSIM: {batch_ssim:.4f}")
                axes[1].axis('off')
                
                # 目标图像
                axes[2].imshow(target_img)
                axes[2].set_title("Target Image")
                axes[2].axis('off')
                
                plt.tight_layout()
                sample_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
                plt.savefig(sample_path, bbox_inches='tight')
                plt.close()
    
    # 计算平均指标
    num_samples = len(metrics) if not max_samples else min(max_samples, len(dataset))
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_time = sum(processing_times) / len(processing_times)
    fps = 1.0 / avg_time
    
    # 打印结果
    print("\n" + "="*50)
    print(f"评估完成，共处理 {num_samples} 张图像")
    print(f"平均 PSNR: {avg_psnr:.4f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均处理时间: {avg_time:.4f} 秒/张")
    print(f"处理速度: {fps:.2f} FPS")
    print("="*50)
    
    # 保存指标结果
    metrics_path = os.path.join(output_dir, "evaluation_metrics.csv")
    with open(metrics_path, 'w') as f:
        f.write("filename,psnr,ssim\n")
        for m in metrics:
            f.write(f"{m['filename']},{m['psnr']},{m['ssim']}\n")
    print(f"已保存详细指标到: {metrics_path}")
    
    # 绘制指标分布图
    plot_metrics_distribution(metrics, output_dir)
    
    return avg_psnr, avg_ssim, metrics

def plot_metrics_distribution(metrics, output_dir):
    """Plot the distribution of PSNR and SSIM metrics across the dataset."""
    psnr_values = [m['psnr'] for m in metrics]
    ssim_values = [m['ssim'] for m in metrics]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PSNR Distribution
    ax1.hist(psnr_values, bins=30, alpha=0.7, color='skyblue')
    ax1.axvline(x=np.mean(psnr_values), color='red', linestyle='dashed', linewidth=2)
    ax1.set_title('PSNR Distribution')
    ax1.set_xlabel('PSNR (dB)')
    ax1.set_ylabel('Number of Images')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # SSIM Distribution
    ax2.hist(ssim_values, bins=30, alpha=0.7, color='lightgreen')
    ax2.axvline(x=np.mean(ssim_values), color='red', linestyle='dashed', linewidth=2)
    ax2.set_title('SSIM Distribution')
    ax2.set_xlabel('SSIM')
    ax2.set_ylabel('Number of Images')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "metrics_distribution.png")
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()
    print(f"Metric distribution plot saved to: {dist_path}")

# ----------------------------
# 主函数
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='低光饱和去模糊模型推理与评估')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='训练好的模型权重文件路径')
    parser.add_argument('--input_path', type=str, default=None, 
                        help='输入图像路径或包含图像的目录 (推理模式使用)')
    parser.add_argument('--data_dir', type=str, default=None, 
                        help='数据集根目录 (评估模式使用, 包含low_blur_noise和high_sharp_scaled子目录)')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='输出目录')
    parser.add_argument('--img_size', type=int, default=256, 
                        help='模型输入图像大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--save_comparison', action='store_true', 
                        help='保存原始图像与增强图像的对比图 (推理模式)')
    parser.add_argument('--evaluate', action='store_true', 
                        help='在整个数据集上评估模型性能')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='评估时最大样本数 (用于测试)')
    parser.add_argument('--save_samples', action='store_true', 
                        help='评估时保存样本结果')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = 'cpu'
    
    print(f"[配置] 使用设备: {args.device}")
    print(f"[配置] 模型路径: {args.model_path}")
    print(f"[配置] 输出目录: {args.output_dir}")
    print(f"[配置] 图像大小: {args.img_size}")
    
    # 加载模型
    model = LowLightDeblurNet()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    print("模型加载成功")
    
    # 确定运行模式
    if args.evaluate:
        # 评估模式 - 在整个数据集上评估模型性能
        if not args.data_dir:
            raise ValueError("评估模式需要指定 --data_dir 参数")
            
        print(f"[评估] 数据集目录: {args.data_dir}")
        print(f"[评估] 最大样本数: {args.max_samples or '无限制'}")
        print(f"[评估] 保存样本: {'是' if args.save_samples else '否'}")
        
        evaluate_on_dataset(
            model, 
            args.device, 
            args.data_dir, 
            img_size=args.img_size,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            save_samples=args.save_samples
        )
    elif args.input_path:
        # 推理模式 - 处理单个图像或目录
        print(f"[推理] 输入路径: {args.input_path}")
        print(f"[推理] 保存对比图: {'是' if args.save_comparison else '否'}")
        
        # 处理输入路径
        if os.path.isdir(args.input_path):
            # 处理目录中的所有图像
            exts = ('*.png', '*.jpg', '*.jpeg')
            image_paths = []
            for ext in exts:
                image_paths += glob(os.path.join(args.input_path, '**', ext), recursive=True)
            
            print(f"找到 {len(image_paths)} 张图像进行增强")
            
            for image_path in image_paths:
                print(f"\n处理图像: {image_path}")
                process_image(
                    model, 
                    args.device, 
                    image_path, 
                    img_size=args.img_size,
                    save_comparison=args.save_comparison,
                    output_dir=args.output_dir
                )
        else:
            # 处理单张图像
            print(f"\n处理图像: {args.input_path}")
            process_image(
                model, 
                args.device, 
                args.input_path, 
                img_size=args.img_size,
                save_comparison=args.save_comparison,
                output_dir=args.output_dir
            )
        
        print("\n处理完成!")
    else:
        print("错误: 请指定运行模式 (--evaluate 或 --input_path)")

if __name__ == '__main__':
    main()