import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 新增FAC模块 (Feature Attention Convolution)
# ----------------------------
class FAC(nn.Module):
    """特征注意力卷积 (来自LEDNet)"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        # 空间注意力
        sa = self.spatial_attention(x)
        # 融合注意力
        return x * ca * sa

# ----------------------------
# 修改后的Retinex分解模块
# ----------------------------
class RetinexDecomposition(nn.Module):
    """Retinex分解模块 (DECM) - 添加FAC和Dropout"""
    def __init__(self, in_channels=3, out_channels=64, dropout_prob=0.1):
        super().__init__()
        self.decompose = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            FAC(32),  # 新增FAC层
            nn.Dropout2d(dropout_prob),  # 新增Dropout
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            FAC(64),  # 新增FAC层
            
            nn.Conv2d(64, out_channels * 2, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        features = self.decompose(x)
        reflectance, illumination = torch.split(features, features.size(1)//2, dim=1)
        reflectance = F.relu(reflectance)
        illumination = torch.sigmoid(illumination)
        return reflectance, illumination

# ----------------------------
# 修改后的InternalRescalingModule
# ----------------------------
class InternalRescalingModule(nn.Module):
    """内部缩放模块 (IRM) - 添加FAC和Dropout"""
    def __init__(self, in_channels, dropout_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fac = FAC(in_channels)  # 新增FAC层
        self.dropout = nn.Dropout2d(dropout_prob)  # 新增Dropout
        
    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = self.fac(x)  # 应用FAC
        x = F.relu(self.conv2(x))
        x = self.dropout(x)  # 应用Dropout
        attn = self.attention(x)
        return identity + x * attn

# ----------------------------
# 修改后的DenseAttentionBlock
# ----------------------------
class DenseAttentionBlock(nn.Module):
    """密集注意力块 (DarkDeblurNet) - 添加FAC和Dropout"""
    def __init__(self, in_channels, dropout_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fac = FAC(in_channels)  # 新增FAC层
        self.dropout = nn.Dropout2d(dropout_prob)  # 新增Dropout
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = self.fac(x1)  # 应用FAC
        x2 = F.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x2 = self.dropout(x2)  # 应用Dropout
        x3 = F.relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        attn = self.attn(x3)
        return x3 * attn

# ----------------------------
# 修改后的ContextualGatingModule
# ----------------------------
class ContextualGatingModule(nn.Module):
    """上下文门控模块 (CGM) - 添加FAC"""
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fac = FAC(in_channels)  # 新增FAC层
        
    def forward(self, x, context):
        context = self.fac(context)  # 应用FAC
        gate_weights = self.gate(context)
        return x * gate_weights

# ----------------------------
# 修改后的LowLightDeblurNet
# ----------------------------
class LowLightDeblurNet(nn.Module):
    """针对低光饱和去模糊的专用模型架构 - 添加FAC和Dropout"""
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128, 256], dropout_prob=0.1):
        super().__init__()
        
        # 1. Retinex分解模块
        self.retinex = RetinexDecomposition(in_channels, features[0], dropout_prob)
        
        # 2. 编码器路径 (处理反射率和光照)
        self.reflectance_encoders = nn.ModuleList()
        self.illumination_encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        ch = features[0]
        for feat in features:
            # 反射率编码器
            self.reflectance_encoders.append(
                nn.Sequential(
                    nn.Conv2d(ch, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    InternalRescalingModule(feat, dropout_prob),  # 添加dropout_prob
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            
            # 光照编码器
            self.illumination_encoders.append(
                nn.Sequential(
                    nn.Conv2d(ch, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    InternalRescalingModule(feat, dropout_prob),  # 添加dropout_prob
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            
            self.pools.append(nn.MaxPool2d(2, 2))
            ch = feat
        
        # 3. Bottleneck (密集注意力)
        self.bottleneck = nn.Sequential(
            DenseAttentionBlock(ch * 2, dropout_prob),  # 添加dropout_prob
            DenseAttentionBlock(ch * 2, dropout_prob),
            DenseAttentionBlock(ch * 2, dropout_prob)
        )
        
        # 4. 解码器路径
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.gating_modules = nn.ModuleList()
        
        current_channels = ch * 2
        
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(current_channels, feat, kernel_size=2, stride=2))
            self.gating_modules.append(ContextualGatingModule(feat * 2))
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(feat * 3, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_prob),  # 新增Dropout层
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = feat
        
        # 5. 输出重建
        self.final_conv = nn.Sequential(
            nn.Conv2d(current_channels, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 1. Retinex分解
        reflectance, illumination = self.retinex(x)
        
        # 编码器路径
        reflectance_skips = []
        illumination_skips = []
        
        for refl_enc, illum_enc, pool in zip(self.reflectance_encoders, 
                                            self.illumination_encoders, 
                                            self.pools):
            reflectance = refl_enc(reflectance)
            reflectance_skips.append(reflectance)
            reflectance = pool(reflectance)
            
            illumination = illum_enc(illumination)
            illumination_skips.append(illumination)
            illumination = pool(illumination)
        
        # 合并特征
        combined = torch.cat([reflectance, illumination], dim=1)
        
        # Bottleneck处理
        x = self.bottleneck(combined)
        
        # 解码器路径
        for up, gate, conv, refl_skip, illum_skip in zip(
            self.ups, 
            self.gating_modules,
            self.up_convs,
            reversed(reflectance_skips),
            reversed(illumination_skips)
        ):
            # 上采样
            x = up(x)
            if x.size() != refl_skip.size():
                x = F.interpolate(x, size=refl_skip.shape[2:], mode='bilinear', align_corners=True)
            
            # 合并跳跃连接
            skip_combined = torch.cat([refl_skip, illum_skip], dim=1)
            
            # 应用上下文门控
            gated = gate(x, skip_combined)
            
            # 拼接特征
            x = torch.cat([skip_combined, gated], dim=1)
            
            # 卷积处理
            x = conv(x)
        
        # 输出重建
        output = self.final_conv(x)
        
        return output



if __name__ == "__main__":
    model = LowLightDeblurNet()
    print(model)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")