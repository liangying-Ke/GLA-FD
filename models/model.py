import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import ConvModule, BlurPool, Attention
from timm.models.layers import DropPath
from timm.models.vision_transformer import trunc_normal_


class FFN(nn.Module):
    def __init__(self, dim_in=3, dim_out=768, stride=1):
        super().__init__()
        self.dim_in = dim_in
        self.conv0 = ConvModule(dim_in   , dim_in   , kernel_size=7, stride=stride, group_mode='DW', mode='CONV-NORM')
        self.conv1 = ConvModule(dim_in   , dim_out*2, kernel_size=1, stride=1     , group_mode=''  , mode='CONV-NORM-ACTV')
        self.conv2 = ConvModule(dim_out*2, dim_out  , kernel_size=1, stride=1     , group_mode=''  , mode='CONV-NORM')
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
 

class PreNorm(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.norm = nn.BatchNorm2d(module.dim_in)

    def forward(self, x):
        return self.norm(self.module(x))


class CouplingEnhancedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = BlurPool(dim) 
        self.conv_h2l = ConvModule(dim, dim, kernel_size=3, group_mode='DW', mode='CONV-NORM-ACTV') 
        self.conv_h2h = ConvModule(dim, dim, kernel_size=3, group_mode='DW', mode='CONV-NORM-ACTV') 
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(dim  , dim*2, kernel_size=1, mode='CONV-NORM-ACTV'),
            ConvModule(dim*2, dim  , kernel_size=1, mode='CONV-NORM'),
            nn.Sigmoid()
        )
        self.proj = ConvModule(dim, dim, kernel_size=3, group_mode='DW', mode='CONV-NORM-ACTV')

    def forward(self, x):
        b, c, h, w = x.shape

        xl = F.interpolate(self.conv_h2l(self.downsample(x)), size=(h, w), mode='bilinear', antialias=True)
        xh = self.conv_h2h(x)
        attn = self.fc(xl + xh)

        xl = xl + xl * attn
        xh = xh + xh * (1-attn)

        xl_1, xl_2 = xl.chunk(2, dim=1)
        xh_1, xh_2 = xh.chunk(2, dim=1)
        return self.proj(torch.cat([xl_1+xh_2, xl_2+xh_1], dim=1))
    

class GlobalLocalInformationBlock(nn.Module):
    def __init__(self, dim_in, use_dual_info=True, use_freq_coupling=True):
        super().__init__()
        self.dim_in = dim_in
        self.use_dual_info = use_dual_info
        self.use_freq_coupling = use_freq_coupling
        
        if use_freq_coupling:
            self.ceb = CouplingEnhancedBlock(dim_in)
            self.proj1 = ConvModule(dim_in, dim_in, kernel_size=3, group_mode='DW', mode='CONV-NORM-ACTV')

        if use_dual_info:
            self.mlp = FFN(dim_in, dim_in)
            self.proj2 = ConvModule(dim_in, dim_in, kernel_size=3, group_mode='DW', mode='CONV-NORM-ACTV')
        self.attn = Attention(dim_in)

    def forward(self, x):
        if self.use_freq_coupling:
            x = x + self.ceb(x)
            x = self.proj1(x)

        if self.use_dual_info:
            x = x + self.attn(x) + self.mlp(x)
            x = self.proj2(x)
        else:
            x = self.attn(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, drop_path=0.2, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim

        self.mlp = PreNorm(FFN(dim, dim))
        self.attn = PreNorm(GlobalLocalInformationBlock(dim))

        self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(x))
        x = x + self.drop_path(self.gamma_2 * self.mlp(x))
        return x


class LinearModule(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.eps = 1e-5
        self.kernel = nn.Parameter(torch.FloatTensor(channels, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    
    def forward(self, embeds):
        out = torch.mm(embeds, self.kernel)

        norms = torch.norm(embeds, p=2, dim=1, keepdim=True)
        kernel_norm = F.normalize(self.kernel, dim=0)
        cos_theta = torch.mm(embeds / norms, kernel_norm)
        
        cos_theta = cos_theta.clamp(-1+self.eps, 1-self.eps) 
        norms = torch.clip(norms, min=0.001, max=100)

        return out, embeds, cos_theta, norms


class Model(nn.Module):
    def __init__(self, num_classes, chs=[64, 128, 256, 512], blks=[1, 1, 3, 1]):
        super().__init__()
        self.pre_dim = chs[0]
        self.stem   = FFN(3, chs[0], stride=1)
        self.stage1 = self.make_layer(BasicBlock, chs[0], blks[0], stride=2)        # 56
        self.stage2 = self.make_layer(BasicBlock, chs[1], blks[1], stride=2)        # 28
        self.stage3 = self.make_layer(BasicBlock, chs[2], blks[2], stride=2)        # 14
        self.stage4 = self.make_layer(BasicBlock, chs[3], blks[3], stride=2)        # 7
        self.head = LinearModule(chs[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if stride == 2:
                layers.append(FFN(self.pre_dim, channels, stride=2))
            layers.append(block(channels))
        self.pre_dim = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)
