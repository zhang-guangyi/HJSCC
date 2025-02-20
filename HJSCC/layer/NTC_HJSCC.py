from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import pickle
from collections import OrderedDict
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torch.distributions as td
import torchvision.transforms.functional as tvf
from compressai.entropy_models import GaussianConditional
from layer import common as common
from layer import HJSCC_EncDec as HJSCC_EncDec
from timm.models.convnext import ConvNeXtBlock
from layer.entropy_coding import gaussian_log_prob_mass

class ConvNeXtBlockAdaLN(nn.Module):
    default_embedding_dim = 256
    def __init__(self, dim, embed_dim=None, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm.affine = False # for FLOPs computing
        # AdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2*dim),
            nn.Unflatten(1, unflattened_size=(1, 1, 2*dim))
        )
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        from timm.layers.mlp import Mlp
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.full(size=(1, out_dim, 1, 1), fill_value=1e-6))
        else:
            self.gamma = None

        self.residual = residual
        self.requires_embedding = True

    def forward(self, x, emb):
        shortcut = x
        # depthwise conv
        x = self.conv_dw(x)
        # layer norm
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        # AdaLN
        embedding = self.embedding_layer(emb)
        shift, scale = torch.chunk(embedding, chunks=2, dim=-1)
        x = x * (1 + scale) + shift
        # MLP
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # scaling
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x



class VDBlock(nn.Module):
    """ Adapted from VDVAE (https://github.com/openai/vdvae)
    - Paper: Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images
    - arxiv: https://arxiv.org/abs/2011.10650
    """
    def __init__(self, in_ch, hidden_ch=None, out_ch=None, residual=True,
                 use_3x3=True, zero_last=False):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or round(in_ch * 0.25)
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.residual = residual
        self.c1 = common.conv_k1s1(in_ch, hidden_ch)
        self.c2 = common.conv_k3s1(hidden_ch, hidden_ch) if use_3x3 else common.conv_k1s1(hidden_ch, hidden_ch)
        self.c3 = common.conv_k3s1(hidden_ch, hidden_ch) if use_3x3 else common.conv_k1s1(hidden_ch, hidden_ch)
        self.c4 = common.conv_k1s1(hidden_ch, out_ch, zero_weights=zero_last)

    def residual_scaling(self, N):
        # This residual scaling improves stability and performance with many layers
        # https://arxiv.org/pdf/2011.10650.pdf, Appendix Table 3
        self.c4.weight.data.mul_(math.sqrt(1 / N))

    def forward(self, x):
        xhat = self.c1(tnf.gelu(x))
        xhat = self.c2(tnf.gelu(xhat))
        xhat = self.c3(tnf.gelu(xhat))
        xhat = self.c4(tnf.gelu(xhat))
        out = (x + xhat) if self.residual else xhat
        return out


class VRLVBlockBase(nn.Module):
    """ Vriable-Rate Latent Variable Block
    """
    default_embedding_dim = 256
    def __init__(self, width, zdim, enc_width, embed_dim=None, kernel_size=7, mlp_ratio=2,flag=0,config=0):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width
        self.config=config
        block = common.ConvNeXtBlockAdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.resnet_front = block(width,   embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = block(width,   embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = block(enc_width, embed_dim, kernel_size=kernel_size)
        self.posterior1 = block(width,     embed_dim, kernel_size=kernel_size)
        self.posterior2 = block(width,     embed_dim, kernel_size=kernel_size)
        self.post_merge = common.conv_k1s1(width + enc_width, width)
        self.posterior  = common.conv_k3s1(width, zdim)
        self.z_proj     = common.conv_k1s1(zdim, width)
        self.prior      = common.conv_k1s1(width, zdim*2)

        self.is_latent_block = True
        self.HJSCC_EncDec = HJSCC_EncDec.HJSCC_EncDec(zdim, flag, self.config)

    def feature_probs_based_Gaussian(self, feature, mean, sigma):
        sigma = sigma.clamp(1e-10, 1e10) if sigma.dtype == torch.float32 else sigma.clamp(1e-10, 1e4)
        gaussian = torch.distributions.normal.Normal(mean, sigma)
        prob = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        likelihoods = torch.clamp(prob, 1e-10, 1e10)  # B C H W
        return likelihoods
    def transform_prior(self, feature, lmb_embedding):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature, lmb_embedding)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, enc_feature, lmb_embedding):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.posterior0(enc_feature, lmb_embedding)
        feature = self.posterior1(feature, lmb_embedding)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged, lmb_embedding)
        qm = self.posterior(merged)
        return qm


    def forward_ntc(self, feature, lmb_embedding, enc_feature=None):
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)
        qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
        z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)  #
        log_prob = gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
        kl = -1.0 * log_prob
        feature_next_ntc = self.resnet_end(feature + self.z_proj(z_sample),lmb_embedding)
        prob = self.feature_probs_based_Gaussian(qm, pm, pv)
        return feature_next_ntc, qm, prob, kl

    def forward_hjscc(self, feature, z_sample, prob, initial_image, scale_h,scale_w, lmb_embedding,snr):
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)
        z_hat, cbr_z = self.HJSCC_EncDec(z_sample, prob, initial_image, scale_h,scale_w,snr)
        feature_next_hjscc = self.resnet_end(feature + self.z_proj(z_hat) , lmb_embedding)

        return feature_next_hjscc, cbr_z




