- # Low Light Image Deblurring

  

  ## 项目概述

  **模型目标**：

  - 在低光饱和环境下，同时实现图像去模糊和增强，提升整体清晰度与细节表现。

  **核心思路**：

  1. **Retinex 分解**：分离反射率与光照，分别优化细节与亮度。
  2. **密集注意力**：在瓶颈层通过多层 DenseAttentionBlock 融合特征，并突出高频细节。
  3. **上下文门控**：在解码器中结合编码特征动态调节信息流，强化重建表现。

  **数据集概况**：

  - 使用 LowBlur 数据集，包含 **10200** 对图像（low_blur_noise 低光模糊 → high_sharp_scaled 清晰高光）。

  **性能指标**：

  - **Checkpoint**: `weights/deblurnet_best.pth`
  - **PSNR**: 34.6055 dB
  - **SSIM**: 0.9463

  **硬件与效率**：

  - **硬件环境**：NVIDIA RTX 4090 24GB ×1
  - **训练效率**：约 8 张图像 / 秒（batch_size=8, img_size=256）
  - **推理速度**：143.89 FPS（以单张图像 256×256 尺寸为准）

  > **注意**：模型在 LowBlur 数据集上表现优秀，但在部分真实拍摄图像中仍存在过饱和和去模糊不足的问题。

  | Dataset  | Link                                                         | Number | Description                                                  |
  | -------- | ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
  | LOL-Blur | [Google Drive / BaiduPan (key: dz6u)](https://pan.baidu.com/s/1CPphxCKQJa_iJAGD6YACuA#list/path=%2F) | 12,000 | 共包含 170 个训练视频和 30 个测试视频，每个视频 60 帧，合计 12,000 对图像。注意：每个视频的前后 30 帧非连续，亮度模拟方式不同。 |

  ## 目录结构

  ```
  LowLightDeblur/
  ├── weights/                    # 模型权重
  │   └── deblurnet_best.pth
  ├── checkpoint/                 # 训练检查点
  ├── dataset/                    # 数据集目录
  │   ├── high_sharp_scaled/      # 清晰高光图像（训练目标）
  │   └── low_blur_noise/         # 低光模糊图像（模型输入）
  ├── result/                     # 训练与测试结果
  │   ├── training_history.png    # 损失与指标曲线
  │   └── test_results/           # 测试输出图像
  ├── test_images/                # 自定义测试样例
  ├── deblur_unet_model.py        # 模型定义
  ├── train_deblur_unet.py        # 训练脚本
  ├── test_deblur_unet.py         # 推理脚本
  └── requirements.txt            # 依赖列表
  ```

  ## 环境依赖

  - Python >= 3.7
  - PyTorch >= 1.8
  - torchvision
  - numpy
  - pillow
  - matplotlib
  - tqdm
  - piq          # 用于计算 PSNR 和 SSIM
  - rawpy （仅在处理 RAW 数据时需要）

  安装依赖：

  ```bash
  pip install -r requirements.txt
  ```

  ## 训练指南

  1. 准备数据：

     - 将低光模糊图像放入 `dataset/low_blur_noise/`
     - 将对应清晰图像放入 `dataset/high_sharp_scaled/`

  2. 配置超参数：在 `train_deblur_unet.py` 中修改批大小、学习率等。

  3. 启动训练：

     ```bash
     python train_deblur_unet.py 
     ```

  4. 模型参数保存在 `/checkpoints`。

  ## 推理/测试与评估指南

  脚本同时支持单图/目录推理模式和全数据集评估模式。

  ### 1. 单图或批量推理模式

  - **参数**: `--input_path`, `--model_path`, `--output_dir`, `--img_size`, `--save_comparison`

  - **示例**: 对 `test_images/` 中所有图像进行推理并保存对比图

    ```bash
    python test_deblur_unet.py \
      --model_path weights/deblurnet_best.pth \
      --input_path test_images/ \
      --output_dir result/test_results/ \
      --img_size 256 \
      --save_comparison
    ```

  - **输出**:

    - `result/test_results/enhanced_*.png`：增强后图像
    - `result/test_results/comparison_*.png`：输入与输出对比图

  ### 2. 全数据集评估模式

  - **参数**: `--evaluate`, `--data_dir`, `--model_path`, `--output_dir`, `--img_size`, `--max_samples`, `--save_samples`

  - **示例**: 在 `dataset/` 下递归评估全部图像，并保存指标及分布图

    ```bash
    python test_deblur_unet.py \
      --model_path weights/deblurnet_best.pth \
      --evaluate \
      --data_dir dataset/ \
      --output_dir result/evaluation_results/ \
      --img_size 256 \
      --save_samples
    ```

  - **输出**:

    - `evaluation_results/evaluation_metrics.csv`：每张图像的 PSNR/SSIM 指标

    - `evaluation_results/metrics_distribution.png`：PSNR/SSIM 分布图

    - 部分样本对比图：`evaluation_results/sample_*.png`

      ![1750822951727](C:\Users\Lenovo\Documents\WeChat Files\wxid_p5azh6fzntre22\FileStorage\Temp\1750822951727.png)

  ## 结果可视化

  - 在 `result/training_history.png` 中查看训练过程中的损失和指标曲线。
  - 在 `result/test_results/` 中查看测试效果。

  