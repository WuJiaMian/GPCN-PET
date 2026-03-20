import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from models.util import wavelet
from models.vmamba import *


class CoordConcat(nn.Module):
    """
    CoordConv操作的实现层。
    它接收一个特征图，并将其与归一化的坐标图在通道维度上拼接。
    """
    def __init__(self, with_r):
        super(CoordConcat, self).__init__()
        # with_r: 一个实验开关，决定是否要额外拼接一个“到中心的半径”通道。
        #         我们先设为False，保持简单。
        self.with_r = with_r

    def forward(self, x):
        B, _, H, W = x.shape
        device = x.device

        # 创建从-1到1的y和x坐标向量
        y_coords_vec = torch.linspace(-1.0, 1.0, steps=H, device=device)
        x_coords_vec = torch.linspace(-1.0, 1.0, steps=W, device=device)

        # 使用广播机制生成二维坐标图
        y_coords = y_coords_vec.view(H, 1).expand(H, W)
        x_coords = x_coords_vec.view(1, W).expand(H, W)

        # 堆叠成 B x 2 x H x W 的坐标图
        coord_map = torch.stack([y_coords, x_coords], dim=0)
        coord_map = coord_map.unsqueeze(0).repeat(B, 1, 1, 1)

        # (可选) 计算并拼接半径通道
        if self.with_r:
            r_coords = torch.sqrt(y_coords**2 + x_coords**2).unsqueeze(0).unsqueeze(0)
            coord_map = torch.cat([coord_map, r_coords.repeat(B, 1, 1, 1)], dim=1)

        # 将坐标图与原始输入特征图在通道维度上拼接
        return torch.cat([x, coord_map], dim=1)




class VmambaBlock(nn.Module):
    """
    Vmamba 的基础块, 用于替换 SwinTransformerBlock.
    输入和输出的 tensor 格式为 (B, H*W, C)，与 Swin Block 保持一致。
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 norm_layer=nn.LayerNorm,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.norm1 = norm_layer(dim)
        # 核心 Vmamba 层 (假设 VSSBlock 是 vmamba.py 中的核心模块)
        # VSSBlock 通常处理 (B, H, W, C) 格式, 所以需要 reshape
        self.mamba_layer = VSSBlock(
            hidden_dim=dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            # forward_type="v0", # 根据你的实现添加
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # Vmamba 块通常不自带 MLP, 但为了与 Swin Block 对齐可以保留一个
        # 根据 VSSM 示例 mlp_ratio=0.0，这里也先移除，如有需要可以加回来
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.0)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # Reshape to (B, H, W, C) for VSSBlock
        x = x.view(B, H, W, C)

        # VSSBlock forward
        x = self.mamba_layer(x)

        # Reshape back to (B, L, C)
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        # 如果需要 MLP
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class RVMB(nn.Module):
    """
    Residual Vmamba Block, 用于替换 RSTB (Residual Swin Transformer Block).
    它包含多个 VmambaBlock。
    """

    def __init__(self, dim, input_resolution, depth,
                 ssm_d_state=16, ssm_ratio=2.0, drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RVMB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VmambaBlock(
                dim=dim,
                input_resolution=input_resolution,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # 这些 PatchEmbed/UnEmbed 主要是为了适配 RSTB 的内部结构,
        # 在 Vmamba 替换中可以简化
        # self.patch_embed = PatchEmbed(...)
        # self.patch_unembed = PatchUnEmbed(...)

    def forward(self, x, x_size):
        shortcut = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)

        # Unpatch (B, L, C) -> (B, C, H, W)
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.conv(x)

        # Patch (B, C, H, W) -> (B, L, C)
        x = x.flatten(2).transpose(1, 2)

        return x + shortcut


# ==================================================================================
# ========================= END: 新增 Vmamba 相关模块 =============================
# ==================================================================================
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, wt_levels, embed_dim, patches_resolution, depths,
                 ssm_d_state, ssm_ratio,  # Vmamba 相关参数
                 drop_rate, dpr, norm_layer, use_checkpoint, img_size, patch_size, resi_connection, bias=True,
                 patch_norm=True, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.patch_norm = patch_norm
        self.wt_levels = wt_levels

        # ==================== 核心修改：高/低频 & 跨层级参数解耦 ====================

        # 1. 为每个小波层级的【低频分量 LL】创建独立的 RVMB 列表
        self.ll_layers = nn.ModuleList()
        # 2. 为每个小波层级的【高频分量 (LH,HL,HH)】创建独立的 RVMB 列表
        self.hf_layers = nn.ModuleList()

        for i in range(self.wt_levels):
            # 每一层的 resolution 会变化
            level_patches_resolution = (
            patches_resolution[0] // (2 ** (i + 1)), patches_resolution[1] // (2 ** (i + 1)))

            # 简化处理，所有层级共享 dpr 列表
            current_dpr = dpr

            # 为当前层级的 LL 分量添加独立的 RVMB
            self.ll_layers.append(
                RVMB(dim=embed_dim,  # LL分量通道数为 embed_dim
                     input_resolution=level_patches_resolution,
                     depth=depths[0],  # 假设所有层级深度相同
                     ssm_d_state=ssm_d_state,
                     ssm_ratio=ssm_ratio,
                     drop_path=current_dpr,
                     norm_layer=norm_layer,
                     use_checkpoint=use_checkpoint,
                     img_size=img_size,
                     patch_size=patch_size,
                     resi_connection=resi_connection
                     )
            )

            # 为当前层级的 HF 分量集合添加独立的 RVMB
            self.hf_layers.append(
                RVMB(dim=embed_dim * 3,  # 3个高频分量通道数是 embed_dim * 3
                     input_resolution=level_patches_resolution,
                     depth=depths[0],  # 假设所有层级深度相同
                     ssm_d_state=ssm_d_state,
                     ssm_ratio=ssm_ratio,
                     drop_path=current_dpr,
                     norm_layer=norm_layer,
                     use_checkpoint=use_checkpoint,
                     img_size=img_size,
                     patch_size=patch_size,
                     resi_connection=resi_connection
                     )
            )

        # 3. 原始的主干路处理 (在小波分解之前的)，仍然保留
        #    这对应于你原始代码中的 self.layer_main，它处理的是最开始的输入
        self.layer_main = RVMB(dim=embed_dim,
                               input_resolution=(patches_resolution[0], patches_resolution[1]),
                               depth=depths[0],
                               ssm_d_state=ssm_d_state,
                               ssm_ratio=ssm_ratio,
                               drop_path=dpr,
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint,
                               img_size=img_size,
                               patch_size=patch_size,
                               resi_connection=resi_connection
                               )
        # =================================================================

        # Patch Embed/Unembed 工具，用于格式转换
        # 为不同通道数的 embed/unembed 创建独立实例
        self.patch_embed_ll = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
                                         embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        self.patch_unembed_ll = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
                                             embed_dim=embed_dim)
        self.patch_embed_hf = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim * 3,
                                         embed_dim=embed_dim * 3, norm_layer=norm_layer if patch_norm else None)
        self.patch_unembed_hf = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim * 3,
                                             embed_dim=embed_dim * 3)
        self.patch_embed_main = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
                                           embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        self.patch_unembed_main = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
                                               embed_dim=embed_dim)

        # 其他部分保持不变
        self.in_channels = in_channels
        self.stride = stride
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet.wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters=self.iwt_filter)
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        if stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        # 1. 主干路处理 (原始逻辑)
        # 这部分处理的是进入 WTConv2d 的原始特征图 x
        x_size_main = (x.shape[2], x.shape[3])
        x_main_patched = self.patch_embed_main(x)
        x_main_processed = self.layer_main(x_main_patched, x_size_main)
        x_main_out = self.patch_unembed_main(x_main_processed, x_size_main)

        # 2. 小波分解与处理
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll_input = x  # 小波分解的输入是原始特征图 x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll_input.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                padded_input = F.pad(curr_x_ll_input, curr_pads)
            else:
                padded_input = curr_x_ll_input

            # 分解为4个分量: LL, LH, HL, HH
            # B, C, 4, H/2, W/2
            decomposed_x = self.wt_function(padded_input)

            # 分离 LL 和 HF (LH, HL, HH)
            ll_comp = decomposed_x[:, :, 0, :, :]  # (B, C, H/2, W/2)
            hf_comp = decomposed_x[:, :, 1:4, :, :]  # (B, C, 3, H/2, W/2)

            # 更新下一次迭代的输入
            curr_x_ll_input = ll_comp

            # ==================== 核心修改：分别处理 LL 和 HF ====================
            # 处理 LL 分量
            x_size_level = (ll_comp.shape[2], ll_comp.shape[3])
            ll_patched = self.patch_embed_ll(ll_comp)
            ll_processed = self.ll_layers[i](ll_patched, x_size_level)
            ll_out = self.patch_unembed_ll(ll_processed, x_size_level)

            # 处理 HF 分量集合
            # 首先 reshape 将 (B, C, 3, H, W) -> (B, C*3, H, W)
            B, C, _, H, W = hf_comp.shape
            hf_reshaped = hf_comp.reshape(B, C * 3, H, W)
            hf_patched = self.patch_embed_hf(hf_reshaped)
            hf_processed = self.hf_layers[i](hf_patched, x_size_level)
            hf_out_reshaped = self.patch_unembed_hf(hf_processed, x_size_level)
            # 再 reshape 回 (B, C, 3, H, W)
            hf_out = hf_out_reshaped.reshape(B, C, 3, H, W)
            # =================================================================

            x_ll_in_levels.append(ll_out)
            x_h_in_levels.append(hf_out)

        # 3. 小波逆变换与融合
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll

        # 最终输出是主干路输出 + 小波路径输出
        x_out = x_main_out + x_tag

        if self.do_stride is not None:
            x_out = self.do_stride(x_out)

        return x_out


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ==================================================================================
# 以下 Swin Transformer 相关的类都不再需要，可以安全删除
# WindowAttention, SwinTransformerBlock, PatchMerging, BasicLayer, RSTB
# ==================================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


# ... [保留 PALHBlock, TransposeLayer, MGWA, 和其他你需要的辅助类] ...
# ... 我将直接从你的原始代码中复制这些类 ...
class PALHBlock(nn.Module):

    def __init__(self, embed_dim,in_channels,nf,out_channels):
        super(PALHBlock, self).__init__()
        nf = 64
        self.coord_concat = CoordConcat(with_r=True)
        self.padm1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=int(nf/2), kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=int(nf/2), out_channels=nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=nf, out_channels=int(nf/2), kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=int(nf/2), out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.padm2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=int(nf/2), kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=int(nf/2), out_channels=nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=nf, out_channels=int(nf/2), kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=int(nf/2), out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
    def forward(self, b):
        b = b.permute(0,2,3,1).contiguous()
        b = torch.view_as_complex(b)
       # b0 = b0.permute(0,2,3,1).contiguous()
       # b0 = torch.view_as_complex(b0)
   #     A0 = torch.abs(b0)
   #   P0 = torch.angle(b0)
        A = torch.abs(b)
        P = torch.angle(b)
        A = A.unsqueeze(1)
        P = P.unsqueeze(1)
        A_with_coords = self.coord_concat(A)  # Shape: B, 3, H, W
        A_processed = self.padm1(A_with_coords)  # Shape: B, 1, H, W
        A1 = A_processed + A
        P1 = self.padm2(P)+P
        P1 = self.padm2(P1)+P1
        #A1 = A1 * (1-mask) + A0.unsqueeze(1) * mask
       # P1 = P1 * (1-mask) + P0.unsqueeze(1) * mask
        R = A1 * torch.cos(P1)
        V = A1 * torch.sin(P1)
        out = torch.complex(R,V)
        out = torch.fft.ifft2(torch.fft.ifftshift(out,dim=(2,3)),dim=(2,3))
        out = torch.view_as_real(out)
        out = out.squeeze(1).permute(0,3,1,2).contiguous()
        return out


class TransposeLayer(nn.Module):
    def __init__(self, dim0, dim1):
        super(TransposeLayer, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


# 注意：你的 MGWA 模块依赖于 WindowAttention，但 WindowAttention 已经被删除了
# 如果 MGWA 模块仍需使用，你需要为它重新定义一个局部的 WindowAttention
# 或者修改 MGWA 使用其他机制。这里我暂时注释掉 MGWA，因为它无法编译。
# class MGWA(nn.Module): ...


class WDPCNet(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6],
                 # 删除了 num_heads, window_size, mlp_ratio
                 # 新增了 Vmamba 参数
                 ssm_d_state=16, ssm_ratio=2.0,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(WDPCNet, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        embed_dim = 72 # 这个参数在 __init__ 中被覆盖了, 使用传入的 embed_dim
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # 删除了 self.window_size

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        # self.mlp_ratio = mlp_ratio # Vmamba 通常不使用独立的 MLP ratio
        self.PMLH = PALHBlock(embed_dim=embed_dim, in_channels=1, nf=num_feat, out_channels=1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 修改 WTconv 的初始化，传入 Vmamba 参数
        self.WTconv = WTConv2d(
            in_channels=int(embed_dim / 3),
            out_channels=int(embed_dim / 3),
            kernel_size=3, stride=1, wt_levels=2,
            embed_dim=int(embed_dim / 3),
            patches_resolution=patches_resolution,
            depths=depths,
            # num_heads=num_heads,  # 移除
            # window_size=window_size, # 移除
            ssm_d_state=ssm_d_state,  # 新增
            ssm_ratio=ssm_ratio,  # 新增
            # qkv_bias=qkv_bias, # 移除
            # qk_scale=qk_scale, # 移除
            drop_rate=drop_rate,
            # attn_drop_rate=attn_drop_rate, # 移除
            dpr=dpr,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )

        self.downconv = nn.Conv2d(embed_dim, int(embed_dim / 3), 3, 1, 1)
        self.upconv = nn.Conv2d(int(embed_dim / 3), embed_dim, 3, 1, 1)

        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(dim // 4, embed_dim, 3, 1, 1))

        # ... [保留 upsampling 和 conv_last 部分] ...
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # Vmamba 对输入尺寸没有严格的整除要求，但为了代码兼容性，暂时保留
        mod_pad_h = (16 - h % 16) % 16  # 使用一个常见的 pad size
        mod_pad_w = (16 - w % 16) % 16
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x, x0):
        x_size = (x.shape[2], x.shape[3])
        xr = x
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.patch_unembed(x, x_size)
        i = 0
        for i in range(1):
            x = self.downconv(x)
           #print(x.shape)
            x = self.WTconv(x)
            x = self.upconv(x)
            x = x + xr

        x = self.patch_embed(x)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def DC(self, x, b, mask):
        x1 = x.permute(0, 2, 3, 1).contiguous()
        x1 = torch.view_as_complex(x1)
        k = torch.fft.fftshift(torch.fft.fftn(x1, dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        k = torch.view_as_real(k)
        k = k.permute(0, 3, 1, 2).contiguous()
        k = k * (1 - mask) + b * mask
        k = k.permute(0, 2, 3, 1).contiguous()
        k = torch.view_as_complex(k)
        x = torch.fft.ifftn(torch.fft.ifftshift(k, dim=(-2, -1)), norm='ortho', dim=(-2, -1))
        x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        H, W = x.shape[2:]  # x (4,1,96,96) (batch_size_in_each_GPU, input_image_channel, H (random-crop 96 in traning and 256 in testing), W)
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        xt = torch.cat((x, torch.zeros_like(x)), dim=1)
        b = xt.permute(0, 2, 3, 1).contiguous()
        b = torch.view_as_complex(b)
        b = torch.fft.fftshift(torch.fft.fftn(b, dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        b = torch.view_as_real(b)
        b = b.permute(0, 3, 1, 2).contiguous()
        for i in range(self.num_layers):
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, x)) + x_first  # res (4,180,96,96)
            x =  self.conv_last(res)+x

            x2 = self.PMLH(b)
            x = torch.cat((x, torch.zeros_like(x)), dim=1) + x2
            b = x.permute(0, 2, 3, 1).contiguous()
            b = torch.view_as_complex(b)
            b = torch.fft.fftshift(torch.fft.fftn(b, dim=(-2, -1), norm='ortho'), dim=(-2, -1))
            b = torch.view_as_real(b)
            b = b.permute(0, 3, 1, 2).contiguous()
            if (i != self.num_layers - 1):
                x = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)


        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 1 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * self.embed_dim * self.embed_dim * 9
        flops += H * W * self.embed_dim * 1 * 9
        return flops

    def params(self):
        params = 0
        params += 1 * self.embed_dim * 9
        params += self.patch_embed.params()
        for i, layer in enumerate(self.layers):
            params += layer.params()
        params += self.embed_dim * self.embed_dim * 9
        params += self.embed_dim * 1 * 9
        return params


if __name__ == '__main__':
    # 1. 实例化网络
    # 为了测试速度，这里可以把 depths 稍微设小一点，您也可以保留默认的 [6, 6, 6, 6]
    print("Instantiating the WDPCNet model...")
    model = WDPCNet(
        img_size=64,
        patch_size=1,
        in_chans=1,
        embed_dim=72,  # 注意：您 __init__ 内部强制写了 embed_dim = 72
        depths=[6, 6,6],
        ssm_d_state=16,
        ssm_ratio=2.0
    )

    # 将模型移动到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 2. 统计可训练参数量 (标准的 PyTorch 统计方式)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.4f} M (Millions)")

    # 3. 测试前向传播 (Forward Pass)
    # 创建一个 dummy input (Batch Size = 2, Channels = 1, H = 64, W = 64)
    dummy_input = torch.randn(2, 1, 64, 64).to(device)
    print(f"Input shape:  {dummy_input.shape}")

    try:
        # 运行前向传播
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!")
    except Exception as e:
        print("Error during forward pass!")
        print(e)