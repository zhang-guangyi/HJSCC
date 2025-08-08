import math
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from layer.layers import Mlp, BasicLayerEnc
import numpy as np

class refine_local(nn.Module):
    def __init__(self,scale_h,scale_w,C):
        super(refine_local, self).__init__()
        self.C=C
        self.input_data_dim=int(scale_h*scale_w*C*2)
        self.output_data_dim=int(scale_h*scale_w*C)
        self.input_index_dim=int(scale_h*scale_w*2)
        self.output_index_dim=int(scale_h*scale_w*2)
        self.pre_process=nn.Linear(self.input_data_dim,self.output_data_dim)
        self.index_process= nn.Sequential(
            nn.GELU(),
            nn.Linear(self.input_index_dim, self.output_index_dim),
        )
        self.data_process=nn.Sequential(
            nn.Linear(self.output_data_dim, 4*self.output_data_dim),
            nn.GELU(),
            nn.Linear(4*self.output_data_dim, self.output_data_dim),
        )

    def forward(self,x,index):
        x=self.pre_process(x)
        N,L=x.size()
        index=self.index_process(index)
        shift, scale = torch.chunk(index, chunks=2, dim=-1)
        N_shift,L_shift=shift.size()
        N_scale,L_scale=shift.size()
        shift = shift.view(N_shift,L_shift,1).repeat(1,1,self.C).view(N,L)
        scale = scale.view(N_scale,L_scale,1).repeat(1,1,self.C).view(N,L)
        data = x * (1 + scale) + shift
        data = self.data_process(data)
        x = x + data
        return x
class RateAdaptionEncoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionEncoder, self).__init__()
        self.C, self.H, self.W = (channel_num, 32, 32)
        self.rate_num = len(rate_choice)
        self.rate_choice = rate_choice
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        print("CONFIG RATE", self.rate_choice_tensor)
        self.weight = nn.Parameter(torch.zeros(self.rate_num , self.C, max(self.rate_choice)))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice)))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.C)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        # trunc_normal_(self.w, std=.02)
        mask = torch.arange(0, max(self.rate_choice)).repeat(self.H * self.W, 1)
        self.register_buffer("mask", mask)

    def forward(self, x, indexes):
        B, C, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        if H != self.H or W != self.W:
            self.update_resolution(H, W, x.get_device())
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, self.C, -1)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, -1)
        mask = self.mask.repeat(B, 1, 1)
        rate_constraint = self.rate_choice_tensor[indexes].reshape(B, H * W, 1).repeat(1, 1, max(self.rate_choice))
        mask_new = torch.zeros_like(mask)
        mask_new[mask < rate_constraint] = 1
        mask_new[mask >= rate_constraint] = 0
        x_BLC_no_mask = torch.matmul(x_BLC.unsqueeze(2), w).squeeze(2) + b
        x_BLC_masked =  x_BLC_no_mask * mask_new
        x_no_mask =  x_BLC_no_mask.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_masked = x_BLC_masked.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        mask_BCHW = mask_new.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x_no_mask, x_masked, mask_BCHW

    def update_resolution(self, H, W, device):
        self.H = H
        self.W = W
        self.num_patches = H * W
        self.mask = torch.arange(0, max(self.rate_choice)).repeat((self.num_patches, 1))
        self.mask = self.mask.to(device)


class JSCCEncoder(nn.Module):
    def __init__(self,zdim, scale_h,scale_w,embed_dim=256, depths=[1, 1, 1], input_resolution=(16, 16),
                 num_heads=[8, 8, 8], window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, rate_choice=[0, 128, 256]):
        super(JSCCEncoder, self).__init__()
        embed_dim=zdim
        self.embed_dim = zdim
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayerEnc(dim=zdim, out_dim=zdim, input_resolution=input_resolution,
                               depth=depths[i_layer], num_heads=num_heads[i_layer],
                               window_size=window_size, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, downsample=None)
            self.layers.append(layer)
        self.rate_adaption = RateAdaptionEncoder(embed_dim, rate_choice)
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, embed_dim))
        trunc_normal_(self.rate_token, std=.02)
        self.refine = Mlp(embed_dim * 2, embed_dim * 8, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.refine_local=refine_local(scale_h,scale_w,zdim)

    def forward(self, x, px, eta,scale_h,scale_w,snr):
        """
        JSCCEncoder encodes latent representations to variable length channel-input vector.

        Arguments:
        x: Latent representation (patch embeddings), shape of BxCxHxW, also viewed as Bx(HxW)xC.
        px: Estimated probability of x, shape of BxCxHxW, also viewed as Bx(HxW)xC.
        eta: Scaling factor from entropy to channel bandwidth cost.

        Returns:
        s_masked: Channel-input vector.
        indexes: The length of each patch embedding, shape of BxHxW.
        mask: Binary mask, shape of BxCxHxW.
        """
        B, C, H, W = x.size()
        hx = torch.clamp_min(-torch.log(px) / math.log(2), 0)
        symbol_num = (torch.sum(hx, dim=1).flatten(0) * eta)
        symbol_num_fact=symbol_num
        indexes_fact = torch.searchsorted(self.rate_choice_tensor, symbol_num_fact).clamp(0, self.rate_num - 1)

        symbol_num_sim=symbol_num.view(B,int(H / scale_h), scale_h, int(W / scale_w), scale_w)
        symbol_num_sim=symbol_num_sim.permute(0, 1, 3, 2, 4)
        symbol_num_sim=symbol_num_sim.contiguous().view(-1, int(scale_h*scale_w))
        symbol_num_sim= torch.sum(symbol_num_sim, dim=1).flatten(0) / scale_h/scale_w
        indexes_sim = torch.searchsorted(self.rate_choice_tensor, symbol_num_sim).clamp(0, self.rate_num - 1)
        indexes_sim = indexes_sim.reshape(-1, 1)
        indexes_sim = indexes_sim.repeat(1, scale_h*scale_w).contiguous().view(B,int(H / scale_h), int(W / scale_w), scale_h, scale_w)
        indexes_sim = indexes_sim.permute(0,1, 3, 2, 4).flatten(0)
        indexes=indexes_sim

        #whole_refine
        x_BLC = x.flatten(2).permute(0, 2, 1)
        px_BLC = px.flatten(2).permute(0, 2, 1)
        whole_refine=self.refine(torch.cat([1 - px_BLC, x_BLC], dim=-1))
        #local_refine
        x_square = x_BLC.view(B,int(H / scale_h), scale_h, int(W / scale_w), scale_w,C)
        x_square = x_square.permute(0, 1, 3, 2, 4, 5)
        x_square = x_square.contiguous().view(-1, int(scale_h * scale_w * C))
        px_square = px_BLC.view(B,int(H / scale_h), scale_h, int(W / scale_w), scale_w,C)
        px_square = px_square.permute(0, 1, 3, 2, 4, 5)
        px_square = px_square.contiguous().view(-1, int(scale_h * scale_w * C))
        indexes_fact=indexes_fact.view(B,int(H / scale_h), scale_h, int(W / scale_w), scale_w).permute(0, 1, 3, 2, 4)
        indexes_fact=indexes_fact.contiguous().view(-1, int(scale_h * scale_w ))
        indexes_sim=indexes_sim.view(B,int(H / scale_h), scale_h, int(W / scale_w), scale_w).permute(0, 1, 3, 2, 4)
        indexes_sim=indexes_sim.contiguous().view(-1, int(scale_h * scale_w ))
        index_info=torch.cat([indexes_fact,indexes_sim],dim=-1).to(torch.float32)
        local_refine = self.refine_local(torch.cat([1 - px_square, x_square], dim=-1),index_info)
        local_refine = local_refine.view(B, int(H / scale_h), int(W / scale_w), scale_h, scale_w,C)
        local_refine = local_refine.permute(0, 1, 3, 2, 4 ,5).contiguous().view(B,H*W,C)

        x_BLC = x_BLC + whole_refine +local_refine
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, C)
        x_BLC = x_BLC + rate_token
        for layer in self.layers:
            x_BLC = layer(x_BLC.contiguous(),snr,H,W)
        x_BLC = self.norm(x_BLC)
        x_BCHW = x_BLC.reshape(B, H, W, C).permute(0, 3, 1, 2)
        s_no_mask, s_masked, mask = self.rate_adaption(x_BCHW, indexes)
        channel_max=self.rate_choice[indexes.max()]
        return s_no_mask, s_masked, mask, indexes,channel_max

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * 2, W * 2)

