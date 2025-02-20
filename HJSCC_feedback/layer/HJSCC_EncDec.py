import numpy as np
import torch
import math
import torch.nn as nn
from loss.distortion import Distortion
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from layer.layers import Mlp
from layer.jscc_encoder import JSCCEncoder
from layer.jscc_decoder import JSCCDecoder
from utils import BCHW2BLN, BLN2BCHW
from channel.channel import Channel
from layer.common import *


class HJSCC_EncDec(nn.Module):
    def __init__(self, zdim, flag, config):
        super().__init__()
        self.config = config
        self.channel = Channel(config)
        self.H=self.W=0
        self.flag=flag
        if flag==0:
            self.fe = JSCCEncoder(zdim,scale_h=1,scale_w=1,**config.fe_kwargs_0)
            self.fd = JSCCDecoder(zdim,**config.fd_kwargs_0)
        elif flag==1:
            self.fe = JSCCEncoder(zdim,scale_h=2,scale_w=2,**config.fe_kwargs_1)
            self.fd = JSCCDecoder(zdim,**config.fd_kwargs_1)
        elif flag==2:
            self.fe = JSCCEncoder(zdim,scale_h=2,scale_w=2,**config.fe_kwargs_2)
            self.fd = JSCCDecoder(zdim,**config.fd_kwargs_2)
        elif flag==3:
            self.fe = JSCCEncoder(zdim,scale_h=4,scale_w=4,**config.fe_kwargs_3)
            self.fd = JSCCDecoder(zdim,**config.fd_kwargs_3)
        self.eta = config.eta


    def feature_probs_based_Gaussian(self, feature, mean, sigma):
        sigma = sigma.clamp(1e-10, 1e10) if sigma.dtype == torch.float32 else sigma.clamp(1e-10, 1e4)
        gaussian = torch.distributions.normal.Normal(mean, sigma)
        prob = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        likelihoods = torch.clamp(prob, 1e-10, 1e10)  # B C H W
        return likelihoods

    def update_resolution(self, H, W):
        # Update attention mask for W-MSA and SW-MSA
        if H != self.H or W != self.W:
            self.fe.update_resolution(H, W)
            self.fd.update_resolution(H, W)
            self.H = H
            self.W = W

    def forward(self, z, z_likelihoods, initial_image, scale_h, scale_w, snr, **kwargs):
        B, C, H, W = z.shape
        B_,C_,H_,W_=initial_image.shape
        self.update_resolution(H, W)
        # DJSCC forward

        s_no_mask, s_masked, mask_BCHW, indexes, channel_max  = self.fe(z, z_likelihoods.detach(), eta=self.eta,scale_h=scale_h,scale_w=scale_w,snr=snr)

        # Pass through the channel.
        mask_BCHW = mask_BCHW.bool()
        channel_input = torch.masked_select(s_masked, mask_BCHW)
        channel_output, channel_usage = self.channel.forward(channel_input,snr)
        s_hat = torch.zeros_like(s_masked)
        s_hat[mask_BCHW] = channel_output
        cbr_z = channel_usage / (C_*H_*W_)
        z_hat = self.fd(s_hat, indexes, snr)
        
        return z_hat, cbr_z
