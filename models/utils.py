import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import DropPath
import numpy as np



class ConvModule(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, groups=1, dilation=1, group_mode='', mode="CONV-NORM-ACTV"):
        self.padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        
        if group_mode == 'DW':
            self.groups = math.gcd(self.dim_in, self.dim_out)

        self.module = nn.Sequential(
            BlurPool(self.dim_in) if stride == 2 else nn.Identity(),
            nn.Conv2d(self.dim_in, self.dim_out, self.kernel_size, 1, self.padding, self.dilation, self.groups, bias=False) if 'CONV' in mode else nn.Identity(),
            nn.BatchNorm2d(self.dim_out) if 'NORM' in mode else nn.Identity(),
            nn.GELU() if 'ACTV' in mode else nn.Identity(),
        )

    def forward(self, x):
        return self.module(x)
        

def get_pad_layer(pad_type):
    if pad_type in ['refl','reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl','replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type=='zero':
        PadLayer = nn.ZeroPad2d
    elif pad_type == 'circular':
        PadLayer = circular_pad
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class GeLu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))
        return x * cdf
    

class circular_pad(nn.Module):
    def __init__(self, padding = (1, 1, 1, 1)):
        super(circular_pad, self).__init__()
        self.pad_sizes = padding
        
    def forward(self, x):
        return F.pad(x, pad = self.pad_sizes , mode = 'circular')
    

class Filter(nn.Module):
    def __init__(self, filt, channels, pad_type=None, pad_sizes=None, scale_l2=False, eps=1e-6):
        super(Filter, self).__init__()
        self.register_buffer('filt', filt[None, None, :, :].repeat((channels, 1, 1, 1)))
        if pad_sizes is not None:
            self.pad = get_pad_layer(pad_type)(pad_sizes)
        else:
            self.pad = None
        self.scale_l2 = scale_l2
        self.eps = eps

    def forward(self, x):
        if self.scale_l2:
            inp_norm = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        if self.pad is not None:
            x = self.pad(x)
        out = F.conv2d(x, self.filt, groups=x.shape[1])
        if self.scale_l2:
            out_norm = torch.norm(out, p=2, dim=(-1, -2), keepdim=True)
            out = out * (inp_norm / (out_norm + self.eps))
        return out


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='zero', filt_size=4, stride=2, pad_off=0, scale_l2=False, eps=1e-6):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.pad_type = pad_type
        self.scale_l2 = scale_l2
        self.eps = eps

        a = self.get_rect(self.filt_size)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.filt = Filter(filt, channels, pad_type, self.pad_sizes, scale_l2)
        if self.filt_size == 1 and self.pad_off == 0:
            self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return self.filt(inp)[:, :, ::self.stride, ::self.stride]

    @staticmethod
    def get_rect(filt_size):
        if filt_size == 1:
            a = np.array([1., ])
        elif filt_size == 2:
            a = np.array([1., 1.])
        elif filt_size == 3:
            a = np.array([1., 2., 1.])
        elif filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        return a
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, ratio=1.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.dim = dim
        self.dim_ratio = int(dim*ratio)
        self.key_dim = self.dim_ratio // num_heads
        self.scale = self.key_dim ** -0.5

        self.num_heads = num_heads
        self.window_size = 7      # windows所具有的pixels大小

        self.qkv        = ConvModule(dim           , self.dim_ratio*3, kernel_size=3, group_mode='DW', mode='CONV-NORM-ACTV')
        self.local_conv = ConvModule(self.dim_ratio, self.dim_ratio  , kernel_size=3, group_mode='DW', mode='CONV')
        self.proj       = ConvModule(self.dim_ratio, dim             , kernel_size=3, group_mode='DW', mode='CONV-NORM-ACTV')

        rel_index_coords = self.double_step_seq(2*self.window_size-1, self.window_size, 1, self.window_size)
        self.rel_position_index = rel_index_coords + rel_index_coords.T
        self.rel_position_index = self.rel_position_index.flip(1).contiguous()
    
        self.rel_position_bias_table = nn.Parameter(torch.zeros((2*self.window_size-1) * (2*self.window_size-1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.rel_position_bias_table, std=.02)

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def window_partition(self, x):
        B, C, H, W = x.shape
        self.nH = H // self.window_size
        self.nW = W // self.window_size
        x = x.permute(0, 2, 3, 1)                                                                       # B, H, W, C 
        x = x.reshape(B, self.nH, self.window_size, self.nW, self.window_size, C).transpose(2, 3)       # B, nH, nW, H', W', C  
        x = x.reshape(B * self.nH * self.nW, self.window_size, self.window_size, C)                     # B*nH*nW, H', W', C
        return x.permute(0, 3, 1, 2).contiguous()                                                       # B*nH*nW, C, H', W'

    def window_concatenate(self, x):
        _, C, _, _ = x.shape            # B', C, H', W'
        x = x.permute(0, 2, 3, 1)                   # B', H', W', C
        x = x.reshape(-1, self.nH, self.nW, self.window_size, self.window_size, C).transpose(2, 3)      # B, nH, H', nW, W', C  
        return x.reshape(-1, self.nH*self.window_size, self.nW*self.window_size, C).permute(0, 3, 1, 2).contiguous() # B, C, H, W  

    def attention(self, x):
        B, C, H, W = x.shape
        WS = self.window_size

        # 計算目標影像的query值，以及參考影像的key和value值
        Q, K, V = self.qkv(x).chunk(3, dim=1)                                         # B, C, H, W
        V = V + self.local_conv(V)

        Q = Q.reshape(B, self.num_heads, -1, H*W).transpose(-2, -1)     # B, num_heads, H*W, key_dim
        K = K.reshape(B, self.num_heads, -1, H*W)                       # B, num_heads, key_dim, H*W
        V = V.reshape(B, self.num_heads, -1, H*W).transpose(-2, -1)     # B, num_heads, H*W, key_dim

        attn = (Q @ K) * self.scale                             # B, num_heads, H*W, H*W

        relative_position_bias = self.rel_position_bias_table[self.rel_position_index.view(-1)].view(WS**2, WS**2, -1)  # N, N, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(-2, -1)                      # B, num_heads, key_dim, H*W
        return out.reshape(B, self.dim_ratio, H, W)
    
    def forward(self, x):
        B, C, H, W = x.shape
        if H <= self.window_size and W <= self.window_size:
            x = self.attention(x)
        else:
            x = self.window_partition(x)
            x  = self.attention(x)
            x  = self.window_concatenate(x)
        return self.proj(x)
