

- # Low Light Image Deblurring

  ## 项目概述

  本项目包含两个低光饱和图像去模糊模型，专为面向LOLBlur数据集而设计：

  ### 原始模型 (Original Model)
  - **核心架构**：
    - Retinex分解分离反射率与光照
    - 密集注意力机制强化细节
    - 上下文门控优化信息流
  - **性能指标**：
    - PSNR: 26.3969 dB
    - SSIM: 0.9005
    - 参数量: ≈52.3M
  - **硬件效率**：
    - NVIDIA RTX 4090
    - 训练速度: 8 img/sec (batch=8, size=256)
    - 推理速度: 125.24 FPS
  
  ### 新模型 (Enhanced Model)
  - **架构优化**：
    - 引入FAC(Feature Attention Convolution)层
    - 增加Dropout正则化
    - 优化瓶颈层设计
  - **训练增强**：
    - 周期性测试集评估
    - 改进混合损失函数
    - 自适应学习率调度
  - **预期优势**：
    - 更好的去模糊效果
    - 减少过拟合
    - 更稳定的训练过程
  - **性能指标**：(训练后更新)
    - PSNR: 26.8281dB
    - SSIM: 0.9140
    - 参数量: ≈52.5M
  - **硬件效率**：
    - NVIDIA RTX 4090
    - 训练速度: 8 img/sec (batch=8, size=256)
    - 推理速度: 84.40 FPS
  
  **模型结构对比**：
  | 原始模型 | 新模型 |
  |----------|--------|
  | ![原始模型结构](./fig/model1_structure.png) | ![新模型结构](./fig/model2_structure.png) |
  
  **数据集概况**：
  | 名称      | 链接                                                                 | 数量   | 描述                     |
  |-----------|----------------------------------------------------------------------|--------|--------------------------|
  | LOL-Blur  | [百度网盘](https://pan.baidu.com/s/1CPphxCKQJa_iJAGD6YACuA) (key: dz6u) | 12,000 | 170训练视频+30测试视频 |

  ## 目录结构
  
  ```
  LowLightDeblur/
  ├── weights/                    # 模型权重
  │   ├── deblurnet_best.pth      # 原始模型
  │   └── deblurnet_fac_best.pth  # 新模型
  ├── checkpoint/                 # 训练检查点
  ├── fig/                        # 可视化结果
  ├── dataset/                    # 数据集
  │   ├── train/                  # 训练集 (10,200张)
  │   	├── high_sharp_scaled/  # 目标图片 
  │   	└── low_blur_noise/     # 输入图片 
  │   └── test/                   # 测试集 (1,800张)
  │   	├── high_sharp_scaled/  # 目标图片 
  │   	└── low_blur_noise/     # 输入图片 
  ├── result/                     # 结果目录
  │   ├── training_history_original.png    # 原始模型训练曲线
  │   ├── training_history_new.png         # 新模型训练曲线
  │   ├── test_results/           # 测试输出
  │   └── evaluation_results/     # 评估结果
  ├── test_images/                # 测试样例
  ├── deblur_unet_model.py        # 原始模型定义
  ├── new_unet_model.py           # 新模型定义
  ├── train_deblur_unet.py        # 原始模型训练
  ├── train_new_unet.py           # 新模型训练
  ├── test.py                     # 统一测试脚本
  └── requirements.txt            # 依赖
  ```
  
  
  
  ## 环境配置
  
  python>=3.7
  
  torch>=1.8.0
  torchvision
  numpy
  pillow
  matplotlib
  tqdm
  lpips
  piq
  rawpy    
  
  ```bash
  pip install -r requirements.txt
  ```
  
  ## 训练指南
  
  ### 原始模型训练
  ```bash
  python train_deblur_unet.py \
    --data_dir dataset/train \
    --ckpt_dir checkpoint \
    --result_dir result
  ```
  
  ### 新模型训练
  ```bash
  python train_new_unet.py \
    --data_dir dataset/train \
    --test_dir dataset/test \
    --ckpt_dir new_checkpoint \
    --result_dir result \
    --test_interval 10  # 每10轮测试一次
  ```
  
  ## 测试与评估
  
  ### 单图或批量推理
  
  - 选择原始模型： --model_type = original，     --model_path = deblurnet_best.pth
  - 选择新模型：     --model_type = enhanced，--model_path = deblurnet_fac_best.pth
  
  ```bash
  # 单图推理，使用原始模型
  python test.py \
    --model_type original \
    --model_path weights/deblurnet_best.pth \
    --input_path test_images/0010.png \
    --output_dir result/test_results/original \
    --img_size 256 \
    --save_comparison
    
  # 批量推理，使用新模型
  python test.py \
    --model_type enhanced \
    --model_path weights/deblurnet_fac_best.pth \
    --input_path test_images/ \
    --output_dir result/test_results/enhanced \
    --img_size 256 \
    --save_comparison
  ```
  
  ### 数据集评估
  
  - 选择原始模型： --model_type = original，    --model_path = deblurnet_best.pth
  - 选择新模型：     --model_type = enhanced，--model_path = deblurnet_fac_best.pth
  
  ```bash
  # 测试集评估，原始模型
  python test.py   --model_type original      --model_path weights/deblurnet_best.pth   --evaluate  --data_dir dataset/test  --output_dir result/evaluation/original --save_samples
  
  # 测试集评估，新模型
  python test.py   --model_type enhanced   --model_path weights/deblurnet_fac_best.pth  --evaluate  --data_dir dataset/test  --output_dir result/evaluation/enhanced --save_samples
  ```
  
  ## 结果可视化
  
  ### 训练过程
  | 原始模型                                                    | 新模型                                               |
  | ----------------------------------------------------------- | ---------------------------------------------------- |
  | ![原始模型训练曲线](./result/training_history_original.png) | ![新模型训练曲线](./result/training_history_new.png) |
  
  ### 测试结果
  | 原始模型                                                    | 新模型                                               |
  | ----------------------------------------------------------- | ---------------------------------------------------- |
  | ![原始模型测试结果](./fig/original_test_set_evaluation.png) | ![新模型测试结果](./fig/new_test_set_evaluation.png) |
  
  ### 指标分布（PSNR和SSIM直方图）
  | 原始模型                                              | 新模型                                             |
  | ----------------------------------------------------- | -------------------------------------------------- |
  | ![PSNR分布](./result/metrics_distribution_origin.png) | ![PSNR分布](./result/metrics_distribution_new.png) |
  
  
  
  ## 性能对比
  
  | 指标          | 原始模型 | 新模型  |
  | ------------- | -------- | ------- |
  | PSNR (dB)     | 26.3969  | 26.8281 |
  | SSIM          | 0.9009   | 0.9140  |
  | LPIPS         | 0.1206   | 0.0954  |
  | 推理速度(FPS) | 125.24   | 84.40   |
