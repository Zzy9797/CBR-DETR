import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union, List
from einops import rearrange, reduce
from collections import OrderedDict


from ultralytics.utils.torch_utils import fuse_conv_and_bn, make_divisible
from timm.layers import CondConv2d, DropPath, trunc_normal_, use_fused_attn, to_2tuple

__all__ = [
           'C3_DCNv2_Dynamic', 'C2f_DCNv2_Dynamic', 'BasicBlock_DCNv2_Dynamic', 'BottleNeck_DCNv2_Dynamic',
           'BasicBlock_Attention', 'BottleNeck_Attention', 'HGBlock_Attention', 'C3_Attention', 'C2f_Attention',
          'SPDConv', 'BasicBlock_WTConv', 'BottleNeck_WTConv',
           'C3_DySnakeConv', 'C2f_DySnakeConv', 'BasicBlock_DySnakeConv', 'BottleNeck_DySnakeConv','SDI',
           'BottleNeck_PConv_Rep', 'BasicBlock_Faster_Block', 'BottleNeck_Faster_Block',
           'BasicBlock_Faster_Block_EMA', 'BottleNeck_Faster_Block_EMA', 'BasicBlock_Faster_Block_Rep', 'BottleNeck_Faster_Block_Rep', 'BasicBlock_Faster_Block_Rep_EMA', 'BottleNeck_Faster_Block_Rep_EMA',
           'ELA_HSFPN', 'CA_HSFPN', 'Multiply', 'DySample', 'CARAFE', 'HWD',      
           'C3_VSS', 'C2f_VSS', 'C3_LVMB', 'C2f_LVMB', 'BasicBlock_VSS', 'BottleNeck_VSS','Fusion',
           'C2f_SAVSS', 'C2f_MambaOut', 'C2f_EfficientVIM', 'C2f_EfficientVIM_CGLU', 'EUCB_SC', 'CSP_MSCB_SC', 'C2f_MambaOut_UniRepLK', 'CrossAttentionBlock', 'C2f_IEL', 'IELC3', 'C2f_RCB',
            'CARM_Block', 'CARM_Node','SPDConv_test'
           ]

class Bottleneck_DCNV3(Bottleneck):
    """Standard bottleneck with DCNV3."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNV3_YOLO(c_, c2, k[1])


class BasicBlock_DCNv3(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DCNV3_YOLO(ch_out, ch_out, 3, act=None)

class BottleNeck_DCNv3(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DCNV3_YOLO(ch_out, ch_out, 3, s=stride, act=None)


class iRMB(nn.Module):
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
				 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
				 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
		super().__init__()
		self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
		self.act = Conv.default_act if act else nn.Identity()
		dim_mid = int(dim_in * exp_ratio)
		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
		self.attn_s = attn_s
		if self.attn_s:
			assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
			self.dim_head = dim_head
			self.window_size = window_size
			self.num_head = dim_in // dim_head
			self.scale = self.dim_head ** -0.5
			self.attn_pre = attn_pre
			self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
			self.v = nn.Sequential(
				nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
				self.act
			)
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			if v_proj:
				self.v = nn.Sequential(
					nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
					self.act
				)
			else:
				self.v = nn.Identity()
		self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
		self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()
		
		self.proj_drop = nn.Dropout(drop)
		self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		B, C, H, W = x.shape
		if self.attn_s:
			# padding
			if self.window_size <= 0:
				window_size_W, window_size_H = W, H
			else:
				window_size_W, window_size_H = self.window_size, self.window_size
			pad_l, pad_t = 0, 0
			pad_r = (window_size_W - W % window_size_W) % window_size_W
			pad_b = (window_size_H - H % window_size_H) % window_size_H
			x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
			n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
			x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
			# attention
			b, c, h, w = x.shape
			qk = self.qk(x)
			qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
			q, k = qk[0], qk[1]
			attn_spa = (q @ k.transpose(-2, -1)) * self.scale
			attn_spa = attn_spa.softmax(dim=-1)
			attn_spa = self.attn_drop(attn_spa)
			if self.attn_pre:
				x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ x
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
				x_spa = self.v(x_spa)
			else:
				v = self.v(x)
				v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ v
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
			# unpadding
			x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
			if pad_r > 0 or pad_b > 0:
				x = x[:, :, :H, :W].contiguous()
		else:
			x = self.v(x)

		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
		
		x = self.proj_drop(x)
		x = self.proj(x)
		
		x = (shortcut + self.drop_path(x)) if self.has_skip else x
		return x

class iRMB_Cascaded(nn.Module):
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
				 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, num_head=16, se_ratio=0.0,
				 attn_s=True, qkv_bias=False, drop=0., drop_path=0., v_group=False):
		super().__init__()
		self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
		self.act = Conv.default_act if act else nn.Identity()
		dim_mid = int(dim_in * exp_ratio)
		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
		self.attn_s = attn_s
		self.num_head = num_head
		if self.attn_s:
			self.attn = LocalWindowAttention(dim_mid)
		else:
			if v_proj:
				self.v = nn.Sequential(
					nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
					self.act
				)
			else:
				self.v = nn.Identity()
		self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
		self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()
		
		self.proj_drop = nn.Dropout(drop)
		self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		B, C, H, W = x.shape
		if self.attn_s:
			x = self.attn(x)
		else:
			x = self.v(x)

		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
		
		x = self.proj_drop(x)
		x = self.proj(x)
		
		x = (shortcut + self.drop_path(x)) if self.has_skip else x
		return x

class BasicBlock_Attention(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)
        
        # self.attention = CoordAtt(ch_out)
        # self.attention = BiLevelRoutingAttention_nchw(ch_out)
        # self.attention = ELA(ch_out)
        # self.attention = SimAM()
        self.attention = AFGCAttention(ch_out)
        
    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.attention(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out

class BottleNeck_Attention(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)
        
        self.attention = CoordAtt(ch_out * self.expansion)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)
        out = self.attention(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param=5, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),nn.BatchNorm2d(outc),nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x,p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0,base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number >  0:
            mod_p_n_x,mod_p_n_y = torch.meshgrid(
                torch.arange(row_number,row_number+1),
                torch.arange(0,mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x,p_n_y  = torch.cat((p_n_x,mod_p_n_x)),torch.cat((p_n_y,mod_p_n_y))
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

class Bottleneck_AKConv(Bottleneck):
    """Standard bottleneck with FocusedLinearAttention."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        # if k[0] == 3:
        #     self.cv1 = AKConv(c1, c2, k[0])
        self.cv2 = AKConv(c2, c2, k[1])

class C3_AKConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_AKConv(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_AKConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_AKConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class BasicBlock_AKConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2a = AKConv(ch_in, ch_out, stride=stride)
        self.branch2b = AKConv(ch_out, ch_out)

class BottleNeck_AKConv(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = AKConv(ch_out, ch_out, stride=stride)

class Bottleneck_RFAConv(Bottleneck):
    """Standard bottleneck with RFAConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        if k[0] == 3:
            self.cv1 = RFAConv(c1, c_, k[0])
        self.cv2 = RFAConv(c_, c2, k[1])

class C3_RFAConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RFAConv(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_RFAConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_RFAConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class BasicBlock_RFAConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2a = RFAConv(ch_in, ch_out, 3, stride=stride)
        self.branch2b = RFAConv(ch_out, ch_out, 3)

class BottleNeck_RFAConv(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = RFAConv(ch_out, ch_out, 3, stride=stride)

class Bottleneck_RFCBAMConv(Bottleneck):
    """Standard bottleneck with RFCBAMConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFCBAMConv(c_, c2, k[1])

class C3_RFCBAMConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RFCBAMConv(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_RFCBAMConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_RFCBAMConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class BasicBlock_RFCBAMConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2a = RFCBAMConv(ch_in, ch_out, 3, stride=stride)
        self.branch2b = RFCBAMConv(ch_out, ch_out, 3)

class BottleNeck_RFCBAMConv(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = RFCBAMConv(ch_out, ch_out, 3, stride=stride)

class Bottleneck_RFCAConv(Bottleneck):
    """Standard bottleneck with RFCBAMConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFCAConv(c_, c2, k[1])

class C3_RFCAConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RFCAConv(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_RFCAConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_RFCAConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class BasicBlock_RFCAConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2a = RFCAConv(ch_in, ch_out, 3, stride=stride)
        self.branch2b = RFCAConv(ch_out, ch_out, 3)

class BottleNeck_RFCAConv(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = RFCAConv(ch_out, ch_out, 3, stride=stride)


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)

class asf_channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(asf_channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
class asf_local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(asf_local_att, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        _, _, h, w = x.size()
        
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)
 
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

class asf_attention_model(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = asf_channel_att(ch)
        self.local_att = asf_local_att(ch)
    def forward(self, x):
        input1,input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x


class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[-1]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.reshape(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)



class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale*k_up)**2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
                                padding=k_up//2*scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)                         # b * 25 * h_ * w_

        X = self.upsmp(X)                               # b * c * h_ * w_
        X = self.unfold(X)                              # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])    # b * c * h_ * w_
        return X

class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI', 'CARM', 'caw_bifpn_dw']
        self.fusion = fusion
        
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'CARM':
            # 确保所有输入通道数一致
            assert len(set(inc_list)) == 1, "CARM requires all inputs to have the same number of channels"
            channels = inc_list[0]
            self.carm_node = CARM_Node(len(inc_list), channels, use_depthwise=True)
            
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        elif self.fusion == 'caw_bifpn_dw':
            # 确保所有输入通道数一致
            assert len(set(inc_list)) == 1, "This BiFPN node requires all inputs to have the same number of channels"
            channels = inc_list[0]
            
            # 实例化我们新的高性价比节点
            self.bifpn_node = CAW_BiFPN_Node_DW(
                num_inputs=len(inc_list), 
                channels=channels
            )
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
        

    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'CARM':
            return self.carm_node(x)
        elif self.fusion == 'SDI':
            return self.SDI(x)
        elif self.fusion == 'caw_bifpn_dw':
            return self.bifpn_node(x)

class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n= int(nOut/2)
        self.conv1x1 = Conv(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = nn.Conv2d(n, n, 3, padding=1, groups=n)
        self.F_sur = nn.Conv2d(n, n, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=n) # surrounding context
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(nOut),
            Conv.default_act
        )
        self.add = add
        self.F_glo= FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = torch.cat([loc, sur], 1) 

        joi_feat = self.bn_act(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output  = input + output
        return output

class RGCSPELAN(nn.Module):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(RGCSPELAN, self).__init__()
        
        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)
        
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        
        self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)
        
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.extend(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))


def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class SBA(nn.Module):

    def __init__(self, inc, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(input_dim//2, input_dim//2, 1)
        self.d_in2 = Conv(input_dim//2, input_dim//2, 1)       
                
        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim//2, kernel_size=1, bias=False)
        
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        
        g_L_feature =  self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature, size= L_feature.size()[2:], align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature, size= H_feature.size()[2:], align_corners=False) 
        
        H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out



class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

class SPDConv_test(nn.Module):

    def __init__(self, inc, ouc, k=1): # 默认卷积核 k=1
        super().__init__()
        # The subsequent convolution layer. Using k=1 is standard and efficient.
        # It processes a tensor with 4 times the input channels.
        self.conv = Conv(inc * 4, ouc, k=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using a more efficient view-permute rearrangement.
        """
        # Get input dimensions
        b, c, h, w = x.shape
        # Calculate output spatial dimensions
        h2, w2 = h // 2, w // 2

        # Space-to-Depth operation
        x = x.view(b, c, h2, 2, w2, 2)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.contiguous().view(b, c * 4, h2, w2)
        
        # Apply the convolution
        x = self.conv(x)
        return x

class FasterFDConv(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_FDConv(
            dim,
            n_div,
            pconv_fw_type
        )
        
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class Bottleneck_PFDConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Partial_FDConv(c1)
        self.cv2 = Partial_FDConv(c2)


class CARM_Block(nn.Module):
    
    def __init__(self, channels, use_depthwise=True):
        super().__init__()
        
        # 主干路径：深度可分离卷积
        if use_depthwise:
            self.conv = nn.Sequential(
                # 深度卷积
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, 
                         groups=channels, bias=False),
                # 点卷积
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU()
            )
        else:
            self.conv = Conv(channels, channels, 3, 1, act=True)
        
        # 并行 EMA 分支
        self.ema = EMA(channels)
        

    def forward(self, x):
        # 保存恒等连接
        identity = x
        
        # 主干路径
        main_path_out = self.conv(x)
        
        # 并行 EMA 分支
        ema_path_out = self.ema(x)
        
        # 第一次加法融合：主干 + EMA
        fused = main_path_out + ema_path_out
        
        # 第二次加法融合：残差连接
        output = identity + fused
        
        return output

class CARM_Node(nn.Module):
    
    def __init__(self, num_inputs, channels, use_depthwise=True):
        super().__init__()
        self.epsilon = 1e-4
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(
            torch.ones(num_inputs, dtype=torch.float32), 
            requires_grad=True
        )
        self.relu = nn.ReLU()
        
        # EMA 精炼块
        self.refinement_block = CARM_Block(channels, use_depthwise)
        
    def forward(self, inputs):
        """
        Args:
            inputs (List[Tensor]): List of input feature maps with same spatial dimensions
        
        Returns:
            Tensor: Fused and refined feature map
        """
        # 快速归一化融合
        weights = self.relu(self.fusion_weight.clone())
        normalized_weights = weights / (torch.sum(weights, dim=0) + self.epsilon)
        
        # 加权求和
        weighted_sum = torch.zeros_like(inputs[0])
        for i, feature_map in enumerate(inputs):
            weighted_sum += normalized_weights[i] * feature_map
        
        # EMA 精炼
        output = self.refinement_block(weighted_sum)
        
        return output




class SpatialAttentionGate(nn.Module):
    """空间注意力门控机制，用于生成内容感知的空间权重"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_att(self.avg_pool(x))
        max_out = self.channel_att(self.max_pool(x))
        channel_att = avg_out + max_out
        
        # 空间注意力
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_spatial, max_spatial], dim=1))
        
        return x * channel_att * spatial_att


class FocusedMultiScaleProcessor(nn.Module):
    """多尺度特征处理模块"""
    def __init__(self, channels, kernel_sizes=(3, 5, 7, 9), reduction=0.5):
        super().__init__()
        hidden_channels = int(channels * reduction)
        
        # 特征压缩
        self.compress = Conv(channels, hidden_channels, 1, act=True)
        
        # 多核并行深度卷积
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, k, padding=k//2, 
                     groups=hidden_channels, bias=False)
            for k in kernel_sizes
        ])
        
        # 特征融合与增强
        self.fusion_conv = Conv(hidden_channels * (len(kernel_sizes) + 1), channels, 1, act=True)
        
        # 残差连接的权重
        self.alpha = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # 压缩特征
        compressed = self.compress(x)
        
        # 多尺度特征提取
        features = [compressed]  # 包含原始压缩特征
        for dw_conv in self.dw_convs:
            features.append(dw_conv(compressed))
        
        # 特征融合
        fused = self.fusion_conv(torch.cat(features, dim=1))
        
        # 加权残差连接
        return x + self.alpha * fused


class ContentAdaptiveWeightGenerator(nn.Module):
    """内容自适应权重生成器"""
    def __init__(self, num_inputs, channels):
        super().__init__()
        self.num_inputs = num_inputs
        
        # 全局上下文提取
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_inputs, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_inputs, 1),
            nn.Sigmoid()
        )
        
        # 局部重要性评估
        self.local_importance = nn.Sequential(
            Conv(channels * num_inputs, channels // 2, 1, act=True),
            Conv(channels // 2, num_inputs, 1, act=False),
            nn.Sigmoid()
        )
        
        # 尺度感知权重调节
        self.scale_modulator = nn.Parameter(torch.ones(num_inputs))
    
    def forward(self, features):
        B, C, H, W = features[0].shape
        
        # 将所有特征图调整到相同尺寸
        unified_features = []
        for feat in features:
            if feat.shape[2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            unified_features.append(feat)
        
        # 拼接所有特征
        concat_features = torch.cat(unified_features, dim=1)
        
        # 生成全局权重
        global_weights = self.global_context(concat_features)  # [B, num_inputs, 1, 1]
        
        # 生成局部权重
        local_weights = self.local_importance(concat_features)  # [B, num_inputs, H, W]
        
        # 结合尺度调节因子
        scale_weights = self.scale_modulator.view(1, -1, 1, 1)
        
        # 最终权重 = 全局权重 × 局部权重 × 尺度权重
        final_weights = global_weights * local_weights * scale_weights
        
        # 归一化权重
        final_weights = final_weights / (torch.sum(final_weights, dim=1, keepdim=True) + 1e-8)
        
        return final_weights

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # groups=in_channels 是深度卷积的核心
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU(inplace=True)
        
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 1x1 卷积是逐点卷积
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        return self.silu(x)
