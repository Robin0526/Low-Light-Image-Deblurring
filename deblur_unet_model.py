import torch
import torch.nn as nn
import torch.nn.functional as F



# ----------------------------
# 定义针对低光饱和去模糊的专用模型架构
# ----------------------------
class RetinexDecomposition(nn.Module):
    """Retinex分解模块 (DECM)"""
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 使用简化的分解结构
        self.decompose = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels * 2, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        features = self.decompose(x)
        # 将特征分为两部分：反射率和光照
        reflectance, illumination = torch.split(features, features.size(1)//2, dim=1)
        reflectance = F.relu(reflectance)
        illumination = torch.sigmoid(illumination)  # 光照应该在0-1之间
        return reflectance, illumination

class InternalRescalingModule(nn.Module):
    """内部缩放模块 (IRM)"""
    def __init__(self, in_channels):
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
        
    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        attn = self.attention(x)
        return identity + x * attn

class DenseAttentionBlock(nn.Module):
    """密集注意力块 (DarkDeblurNet)"""
    def __init__(self, in_channels):
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
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = F.relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        attn = self.attn(x3)
        return x3 * attn

class ContextualGatingModule(nn.Module):
    """上下文门控模块 (CGM)"""
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, context):
        # 计算门控权重
        gate_weights = self.gate(context)
        # 应用门控
        return x * gate_weights

class LowLightDeblurNet(nn.Module):
    """针对低光饱和去模糊的专用模型架构"""
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128, 256]):
        super().__init__()
        
        # 1. Retinex分解模块
        self.retinex = RetinexDecomposition(in_channels, features[0])
        
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
                    InternalRescalingModule(feat),
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            
            # 光照编码器
            self.illumination_encoders.append(
                nn.Sequential(
                    nn.Conv2d(ch, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    InternalRescalingModule(feat),
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            
            self.pools.append(nn.MaxPool2d(2, 2))
            ch = feat
        
        # 3. Bottleneck (密集注意力)
        # 通道数调整为原始设计的2倍
        self.bottleneck = nn.Sequential(
            DenseAttentionBlock(ch * 2),  # 结合反射率和光照特征
            DenseAttentionBlock(ch * 2),
            DenseAttentionBlock(ch * 2)
        )
        
        # 4. 解码器路径
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.gating_modules = nn.ModuleList()
        
        # 初始通道数为瓶颈层输出通道数
        current_channels = ch * 2
        
        for feat in reversed(features):
            # 上采样层
            self.ups.append(nn.ConvTranspose2d(current_channels, feat, kernel_size=2, stride=2))
            # self.ups.append(nn.ConvTranspose2d(current_channels, feat, kernel_size=2, stride=2))
            # 门控模块
            self.gating_modules.append(ContextualGatingModule(feat * 2))
            # 卷积块
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(feat * 3, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = feat  # 更新当前通道数为当前特征数
        
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
        
        # 分别处理反射率和光照
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
