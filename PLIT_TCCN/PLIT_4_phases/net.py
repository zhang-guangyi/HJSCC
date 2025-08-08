from layer.entropy_coding import *
from layer.jscc_encoder import *
from layer.jscc_decoder import *
from layer.layers import *
from layer.PLIT_EncDec import *
from layer.NTC_PLIT import *
from layer.common import *
import torch.nn.functional as tnf
from layer import PLIT_EncDec

def preprocess_input(im: torch.Tensor):
    im_shift = -0.4546259594901961
    im_scale = 3.67572653978347
    x = (im + im_shift) * im_scale
    return x
def process_output(x: torch.Tensor):
    im_hat = x.clone().clamp_(min=-1.0, max=1.0).mul_(0.5).add_(0.5)
    return im_hat
def preprocess_target(im: torch.Tensor):
    x = (im - 0.5) * 2.0
    return x
def pad_divisible_by(img, div=64):
    """ Pad an PIL.Image at right and bottom border \
         such that both sides are divisible by `div`.

    Args:
        img (PIL.Image): image
        div (int, optional): `div`. Defaults to 64.

    Returns:
        PIL.Image: padded image
    """
    B,C, h_old, w_old = img.size()
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_tgt = round(div * math.ceil(h_old / div))
    w_tgt = round(div * math.ceil(w_old / div))
    # left, top, right, bottom
    padding = (0, 0, (w_tgt - w_old), (h_tgt - h_old))
    padded = tvf.pad(img, padding=padding, padding_mode='edge')
    return padded

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_nums = [6, 6, 6, 4, 2]
        dec_nums = [1, 2, 3, 3, 3]
        z_dims = [32, 32, 96, 8]
        im_channels=3
        self.config=config
        self.device=torch.device("cuda")
        ch=96

        self.patch_downsample_1 = patch_downsample(im_channels, 192, rate=4)
        self.ConvNeXtBlockAdaLN_1_1 = ConvNeXtBlockAdaLN(192, kernel_size=7)
        self.ConvNeXtBlockAdaLN_1_2 = ConvNeXtBlockAdaLN(192, kernel_size=7)
        self.ConvNeXtBlockAdaLN_1_3 = ConvNeXtBlockAdaLN(192, kernel_size=7)
        self.ConvNeXtBlockAdaLN_1_4 = ConvNeXtBlockAdaLN(192, kernel_size=7)
        self.ConvNeXtBlockAdaLN_1_5 = ConvNeXtBlockAdaLN(192, kernel_size=7)
        self.ConvNeXtBlockAdaLN_1_6 = ConvNeXtBlockAdaLN(192, kernel_size=7)
        self.ConvNeXtBlockAdaLN_1_7 = ConvNeXtBlockAdaLN(192)
        self.patch_downsample_2 = patch_downsample(192, 384)

        self.ConvNeXtBlockAdaLN_2_1 = ConvNeXtBlockAdaLN(384, kernel_size=7)
        self.ConvNeXtBlockAdaLN_2_2 = ConvNeXtBlockAdaLN(384, kernel_size=7)
        self.ConvNeXtBlockAdaLN_2_3 = ConvNeXtBlockAdaLN(384, kernel_size=7)
        self.ConvNeXtBlockAdaLN_2_4 = ConvNeXtBlockAdaLN(384, kernel_size=7)
        self.ConvNeXtBlockAdaLN_2_5 = ConvNeXtBlockAdaLN(384, kernel_size=7)
        self.ConvNeXtBlockAdaLN_2_6 = ConvNeXtBlockAdaLN(384, kernel_size=7)
        self.ConvNeXtBlockAdaLN_2_7 = ConvNeXtBlockAdaLN(384)
        self.patch_downsample_3 = patch_downsample(384, 512)

        self.ConvNeXtBlockAdaLN_3_1 = ConvNeXtBlockAdaLN(512, kernel_size=5)
        self.ConvNeXtBlockAdaLN_3_2 = ConvNeXtBlockAdaLN(512, kernel_size=5)
        self.ConvNeXtBlockAdaLN_3_3 = ConvNeXtBlockAdaLN(512, kernel_size=5)
        self.ConvNeXtBlockAdaLN_3_4 = ConvNeXtBlockAdaLN(512, kernel_size=5)
        self.ConvNeXtBlockAdaLN_3_5 = ConvNeXtBlockAdaLN(512, kernel_size=5)
        self.ConvNeXtBlockAdaLN_3_6 = ConvNeXtBlockAdaLN(512, kernel_size=5)
        self.ConvNeXtBlockAdaLN_3_7 = ConvNeXtBlockAdaLN(512)
        self.patch_downsample_4 = patch_downsample(512, 512)

        self.ConvNeXtBlockAdaLN_4_1 = ConvNeXtBlockAdaLN(512, kernel_size=3)
        self.ConvNeXtBlockAdaLN_4_2 = ConvNeXtBlockAdaLN(512, kernel_size=3)
        self.ConvNeXtBlockAdaLN_4_3 = ConvNeXtBlockAdaLN(512, kernel_size=3)
        self.ConvNeXtBlockAdaLN_4_4 = ConvNeXtBlockAdaLN(512, kernel_size=3)
        self.ConvNeXtBlockAdaLN_4_5 = ConvNeXtBlockAdaLN(512)
        self.patch_downsample_5 = patch_downsample(512, 512)

        self.ConvNeXtBlockAdaLN_5_1 = ConvNeXtBlockAdaLN(512, kernel_size=1)
        self.ConvNeXtBlockAdaLN_5_2 = ConvNeXtBlockAdaLN(512, kernel_size=1)
        self.ConvNeXtBlockAdaLN_5_3 = ConvNeXtBlockAdaLN(512, kernel_size=1)
        self.ConvNeXtBlockAdaLN_5_4 = ConvNeXtBlockAdaLN(512, kernel_size=1)

        self.compress_1=VRLVBlockBase(512, z_dims[0], enc_width=512, kernel_size=1, mlp_ratio=4,flag=0,config=config)
        self.ConvNeXtBlockAdaLN_6_1 = ConvNeXtBlockAdaLN(512, kernel_size=1, mlp_ratio=4)
        self.patch_upsample_1 = patch_upsample(512, 512, rate=2)

        self.ConvNeXtBlockAdaLN_7_1 = ConvNeXtBlockAdaLN(512, kernel_size=3, mlp_ratio=3)
        self.compress_2_1=VRLVBlockBase(512, z_dims[1], enc_width=512, kernel_size=3, mlp_ratio=3,flag=1,config=config)
        self.compress_2_2=VRLVBlockBase(512, z_dims[1], enc_width=512, kernel_size=3, mlp_ratio=3,flag=1,config=config)
        self.ConvNeXtBlockAdaLN_7_2 = ConvNeXtBlockAdaLN(512, kernel_size=3, mlp_ratio=3)
        self.patch_upsample_2 = patch_upsample(512, 384, rate=2)

        self.ConvNeXtBlockAdaLN_8_1 = ConvNeXtBlockAdaLN(384, kernel_size=5, mlp_ratio=2)
        self.compress_3_1 = VRLVBlockBase(384, z_dims[2],  enc_width=512, kernel_size=5, mlp_ratio=2,flag=2,config=config)
        self.compress_3_2 = VRLVBlockBase(384, z_dims[2],  enc_width=512, kernel_size=5, mlp_ratio=2,flag=2,config=config)
        self.compress_3_3 = VRLVBlockBase(384, z_dims[2],  enc_width=512, kernel_size=5, mlp_ratio=2,flag=2,config=config)
        self.ConvNeXtBlockAdaLN_8_2 = ConvNeXtBlockAdaLN(384, kernel_size=5, mlp_ratio=2)
        self.patch_upsample_3 = patch_upsample(384, 256, rate=2)

        self.ConvNeXtBlockAdaLN_9_1 = ConvNeXtBlockAdaLN(256, kernel_size=7, mlp_ratio=1.75)
        self.compress_4_1 = VRLVBlockBase(256, z_dims[3],  enc_width=384, kernel_size=7, mlp_ratio=1.75,flag=3,config=config)
        self.compress_4_2 = VRLVBlockBase(256, z_dims[3],  enc_width=384, kernel_size=7, mlp_ratio=1.75,flag=3,config=config)
        self.compress_4_3 = VRLVBlockBase(256, z_dims[3],  enc_width=384, kernel_size=7, mlp_ratio=1.75,flag=3,config=config)
        self.ConvNeXtBlockAdaLN_9_2 = ConvNeXtBlockAdaLN(256, kernel_size=7, mlp_ratio=1.75)
        self.patch_upsample_4 = patch_upsample(256, 128, rate=2)

        self.ConvNeXtBlockAdaLN_10_1 = ConvNeXtBlockAdaLN(128, kernel_size=7, mlp_ratio=1.5)
        self.ConvNeXtBlockAdaLN_10_2 = ConvNeXtBlockAdaLN(128, kernel_size=7, mlp_ratio=1.5)
        self.ConvNeXtBlockAdaLN_10_3 = ConvNeXtBlockAdaLN(128, kernel_size=7, mlp_ratio=1.5)
        self.ConvNeXtBlockAdaLN_10_4 = ConvNeXtBlockAdaLN(128, kernel_size=7, mlp_ratio=1.5)
        self.ConvNeXtBlockAdaLN_10_5 = ConvNeXtBlockAdaLN(128, kernel_size=7, mlp_ratio=1.5)
        self.ConvNeXtBlockAdaLN_10_6 = ConvNeXtBlockAdaLN(128, kernel_size=7, mlp_ratio=1.5)
        self.ConvNeXtBlockAdaLN_10_7 = ConvNeXtBlockAdaLN(128, kernel_size=7, mlp_ratio=1.5)
        self.ConvNeXtBlockAdaLN_10_8 = ConvNeXtBlockAdaLN(128, kernel_size=7, mlp_ratio=1.5)
        self.patch_upsample_5 = patch_upsample(128, im_channels, rate=4)

        self.bias = nn.Parameter(torch.zeros(1, 512, 1, 1))
        self.lmb_embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.PLIT_EncDec_1 = PLIT_EncDec.PLIT_EncDec(z_dims[0], flag=0, config = self.config)
        self.PLIT_EncDec_2_1 = PLIT_EncDec.PLIT_EncDec(z_dims[1], flag=1, config = self.config)
        self.PLIT_EncDec_2_2 = PLIT_EncDec.PLIT_EncDec(z_dims[1], flag=1, config = self.config)
        self.PLIT_EncDec_3_1 = PLIT_EncDec.PLIT_EncDec(z_dims[2], flag=2, config = self.config)
        self.PLIT_EncDec_3_2 = PLIT_EncDec.PLIT_EncDec(z_dims[2], flag=2, config = self.config)
        self.PLIT_EncDec_3_3 = PLIT_EncDec.PLIT_EncDec(z_dims[2], flag=2, config = self.config)
        self.PLIT_EncDec_4_1 = PLIT_EncDec.PLIT_EncDec(z_dims[3], flag=3, config = self.config)
        self.PLIT_EncDec_4_2 = PLIT_EncDec.PLIT_EncDec(z_dims[3], flag=3, config = self.config)
        self.PLIT_EncDec_4_3 = PLIT_EncDec.PLIT_EncDec(z_dims[3], flag=3, config = self.config)

    def expand_to_tensor(self, input_, n):
        assert isinstance(input_, (torch.Tensor, float, int)), f'{type(input_):}'
        if isinstance(input_, torch.Tensor) and (input_.numel() == 1):
            input_ = input_.item()
        if isinstance(input_, (float, int)):
            input_ = torch.full(size=(n,), fill_value=float(input_), device=self.device)
        assert input_.shape == (n,), f'{input_:}, {input_.shape:}'
        return input_

    def _lmb_scaling(self, lmb: torch.Tensor):
        # p = 3.0
        # lmb_input = torch.pow(lmb / self.MAX_LMB, 1/p) * self._sin_period
        lmb_input = torch.log(lmb) * 64 / math.log(8192)
        return lmb_input
    def _get_lmb_embedding(self, lmb, n):
        lmb = self.expand_to_tensor(lmb, n=n)
        scaled = self._lmb_scaling(lmb)
        embedding = sinusoidal_embedding(scaled, dim=256,max_period=64)
        embedding = self.lmb_embedding(embedding)
        return embedding
    def sample_lmb(self, n):
        low, high = [16,1024] # original lmb space, 16 to 1024
        p = 3.0
        low, high = math.pow(low, 1/p), math.pow(high, 1/p) # transformed space
        transformed_lmb = low + (high-low) * torch.rand(n, device=self.device)
        lmb = torch.pow(transformed_lmb, exponent=p)
        return lmb

    def forward(self, im, lmb=None,train=1):
        B,C,H,W=im.size()
        initial_image=im
        x=preprocess_input(pad_divisible_by(im, div=64))
        lmb = 128
        emb = self._get_lmb_embedding(lmb, n=im.shape[0])
        weight = 256
        x_target = preprocess_target(im)

        x = self.patch_downsample_1(x)
        x = self.ConvNeXtBlockAdaLN_1_1(x,emb)
        x = self.ConvNeXtBlockAdaLN_1_2(x,emb)
        x = self.ConvNeXtBlockAdaLN_1_3(x,emb)
        x = self.ConvNeXtBlockAdaLN_1_4(x,emb)
        x = self.ConvNeXtBlockAdaLN_1_5(x,emb)
        x = self.ConvNeXtBlockAdaLN_1_6(x,emb)
        x = self.ConvNeXtBlockAdaLN_1_7(x,emb)
        x = self.patch_downsample_2(x)
        x = self.ConvNeXtBlockAdaLN_2_1(x,emb)
        x = self.ConvNeXtBlockAdaLN_2_2(x,emb)
        x = self.ConvNeXtBlockAdaLN_2_3(x,emb)
        x = self.ConvNeXtBlockAdaLN_2_4(x,emb)
        x = self.ConvNeXtBlockAdaLN_2_5(x,emb)
        x = self.ConvNeXtBlockAdaLN_2_6(x,emb)
        feature_4 = x

        x = self.ConvNeXtBlockAdaLN_2_7(x,emb)
        x = self.patch_downsample_3(x)
        x = self.ConvNeXtBlockAdaLN_3_1(x,emb)
        x = self.ConvNeXtBlockAdaLN_3_2(x,emb)
        x = self.ConvNeXtBlockAdaLN_3_3(x,emb)
        x = self.ConvNeXtBlockAdaLN_3_4(x,emb)
        x = self.ConvNeXtBlockAdaLN_3_5(x,emb)
        x = self.ConvNeXtBlockAdaLN_3_6(x,emb)
        feature_3 = x

        x = self.ConvNeXtBlockAdaLN_3_7(x,emb)
        x = self.patch_downsample_4(x)
        x = self.ConvNeXtBlockAdaLN_4_1(x,emb)
        x = self.ConvNeXtBlockAdaLN_4_2(x,emb)
        x = self.ConvNeXtBlockAdaLN_4_3(x,emb)
        x = self.ConvNeXtBlockAdaLN_4_4(x,emb)
        feature_2 = x

        x = self.ConvNeXtBlockAdaLN_4_5(x,emb)
        x = self.patch_downsample_5(x)
        x = self.ConvNeXtBlockAdaLN_5_1(x,emb)
        x = self.ConvNeXtBlockAdaLN_5_2(x,emb)
        x = self.ConvNeXtBlockAdaLN_5_3(x,emb)
        x = self.ConvNeXtBlockAdaLN_5_4(x,emb)
        feature_1 = x


        feature_next_0=self.bias.expand(feature_1.shape)
        feature_next_ntc_1,z_sample_1,prob_1,kl_1 = \
            self.compress_1.forward_ntc(feature=feature_next_0,lmb_embedding=emb,enc_feature=feature_1)
        feature_next_ntc_1 = self.ConvNeXtBlockAdaLN_6_1(feature_next_ntc_1,emb)
        feature_next_ntc_1 = self.patch_upsample_1(feature_next_ntc_1)
        feature_next_ntc_1 = self.ConvNeXtBlockAdaLN_7_1(feature_next_ntc_1,emb)

        feature_next_ntc_2_1,z_sample_2_1,prob_2_1,kl_2_1 = \
            self.compress_2_1.forward_ntc(feature=feature_next_ntc_1,lmb_embedding=emb, enc_feature=feature_2)
        feature_next_ntc_2_2,z_sample_2_2,prob_2_2,kl_2_2 = \
            self.compress_2_2.forward_ntc(feature=feature_next_ntc_2_1,lmb_embedding=emb,enc_feature=feature_2)
        feature_next_ntc_2 = self.ConvNeXtBlockAdaLN_7_2(feature_next_ntc_2_2,emb)
        feature_next_ntc_2 = self.patch_upsample_2(feature_next_ntc_2)
        feature_next_ntc_2 = self.ConvNeXtBlockAdaLN_8_1(feature_next_ntc_2,emb)


        feature_next_ntc_3_1,z_sample_3_1,prob_3_1,kl_3_1 = \
            self.compress_3_1.forward_ntc(feature=feature_next_ntc_2, lmb_embedding=emb,enc_feature=feature_3)
        feature_next_ntc_3_2,z_sample_3_2,prob_3_2,kl_3_2 = \
            self.compress_3_2.forward_ntc(feature=feature_next_ntc_3_1,lmb_embedding=emb, enc_feature=feature_3)
        feature_next_ntc_3_3,z_sample_3_3,prob_3_3,kl_3_3 = \
            self.compress_3_3.forward_ntc(feature=feature_next_ntc_3_2, lmb_embedding=emb,enc_feature=feature_3)
        feature_next_ntc_3 = self.ConvNeXtBlockAdaLN_8_2(feature_next_ntc_3_3, emb)
        feature_next_ntc_3 = self.patch_upsample_3(feature_next_ntc_3)
        feature_next_ntc_3 = self.ConvNeXtBlockAdaLN_9_1(feature_next_ntc_3, emb)

        feature_next_ntc_4_1,z_sample_4_1,prob_4_1,kl_4_1 = \
            self.compress_4_1.forward_ntc(feature=feature_next_ntc_3, lmb_embedding=emb, enc_feature=feature_4)
        feature_next_ntc_4_2,z_sample_4_2,prob_4_2,kl_4_2 = \
            self.compress_4_2.forward_ntc(feature=feature_next_ntc_4_1, lmb_embedding=emb,enc_feature=feature_4)
        feature_next_ntc_4_3,z_sample_4_3,prob_4_3,kl_4_3 = \
            self.compress_4_3.forward_ntc(feature=feature_next_ntc_4_2, lmb_embedding=emb, enc_feature=feature_4)
        feature_next_ntc_4 = self.ConvNeXtBlockAdaLN_9_2(feature_next_ntc_4_3, emb)
        feature_next_ntc_4 = self.patch_upsample_4(feature_next_ntc_4)
        feature_next_ntc_4 = self.ConvNeXtBlockAdaLN_10_1(feature_next_ntc_4, emb)

        feature_next_ntc_5 = self.ConvNeXtBlockAdaLN_10_2(feature_next_ntc_4, emb)
        feature_next_ntc_5 = self.ConvNeXtBlockAdaLN_10_3(feature_next_ntc_5, emb)
        feature_next_ntc_5 = self.ConvNeXtBlockAdaLN_10_4(feature_next_ntc_5, emb)
        feature_next_ntc_5 = self.ConvNeXtBlockAdaLN_10_5(feature_next_ntc_5, emb)
        feature_next_ntc_5 = self.ConvNeXtBlockAdaLN_10_6(feature_next_ntc_5, emb)
        feature_next_ntc_5 = self.ConvNeXtBlockAdaLN_10_7(feature_next_ntc_5, emb)
        feature_next_ntc_5 = self.ConvNeXtBlockAdaLN_10_8(feature_next_ntc_5, emb)
        feature_next_ntc_5 = self.patch_upsample_5(feature_next_ntc_5)
        feature_next_ntc_5_nopad=feature_next_ntc_5[:,:,:H,:W]

        mse_ntc = tnf.mse_loss(feature_next_ntc_5_nopad, x_target, reduction='none').mean(dim=(1, 2, 3))
        out_loss_ntc = mse_ntc * weight
        x_hat_ntc=feature_next_ntc_5_nopad
        im_hat_ntc = process_output(x_hat_ntc.detach())
        im_hat_ntc_nopad = im_hat_ntc[:,:,:H,:W]
        im_mse_ntc = tnf.mse_loss(im_hat_ntc_nopad, im, reduction='mean')
        psnr_ntc = -10 * math.log10(im_mse_ntc.item())

        if train==0:
            snr = self.config.channel['chan_param']
        else:
            snr = self.config.channel['chan_param']
        z_hat_1, cbr_z_1 = self.PLIT_EncDec_1(z_sample_1, prob_1, initial_image, scale_h=1, scale_w=1, train=train,snr=snr)  # plit对应zi
        z_hat_2_1, cbr_z_2_1 = self.PLIT_EncDec_2_1(z_sample_2_1, prob_2_1, initial_image, scale_h=2, scale_w=2, train=train,snr=snr)  # plit对应zi
        z_hat_2_2, cbr_z_2_2 = self.PLIT_EncDec_2_2(z_sample_2_2, prob_2_2, initial_image, scale_h=2, scale_w=2, train=train,snr=snr)  # plit对应zi
        z_hat_3_1, cbr_z_3_1 = self.PLIT_EncDec_3_1(z_sample_3_1, prob_3_1, initial_image, scale_h=2, scale_w=2, train=train,snr=snr)  # plit对应zi
        z_hat_3_2, cbr_z_3_2 = self.PLIT_EncDec_3_2(z_sample_3_2, prob_3_2, initial_image, scale_h=2, scale_w=2, train=train,snr=snr)  # plit对应zi
        z_hat_3_3, cbr_z_3_3 = self.PLIT_EncDec_3_3(z_sample_3_3, prob_3_3, initial_image, scale_h=2, scale_w=2, train=train,snr=snr)  # plit对应zi
        z_hat_4_1, cbr_z_4_1 = self.PLIT_EncDec_4_1(z_sample_4_1, prob_4_1, initial_image, scale_h=4, scale_w=4, train=train,snr=snr)  # plit对应zi
        z_hat_4_2, cbr_z_4_2 = self.PLIT_EncDec_4_2(z_sample_4_2, prob_4_2, initial_image, scale_h=4, scale_w=4, train=train,snr=snr)  # plit对应zi
        z_hat_4_3, cbr_z_4_3 = self.PLIT_EncDec_4_3(z_sample_4_3, prob_4_3, initial_image, scale_h=4, scale_w=4, train=train,snr=snr)  # plit对应zi

        #node=3
        feature_next_plit_1,_ = \
            self.compress_1.forward_plit(feature=feature_next_0,z_hat=z_hat_1,cbr_z=cbr_z_1,lmb_embedding=emb,block=1,threshold=1)
        feature_next_plit_1 = self.ConvNeXtBlockAdaLN_6_1(feature_next_plit_1,emb)
        feature_next_plit_1 = self.patch_upsample_1(feature_next_plit_1)
        feature_next_plit_1 = self.ConvNeXtBlockAdaLN_7_1(feature_next_plit_1,emb)

        feature_next_plit_2_1,_ = \
            self.compress_2_1.forward_plit(feature=feature_next_plit_1,z_hat=z_hat_2_1,cbr_z=cbr_z_2_1,lmb_embedding=emb,block=1,threshold=2)
        feature_next_plit_2_2,_= \
            self.compress_2_2.forward_plit(feature=feature_next_plit_2_1,z_hat=z_hat_2_2,cbr_z=cbr_z_2_2,lmb_embedding=emb,block=1,threshold=3)
        feature_next_plit_2 = self.ConvNeXtBlockAdaLN_7_2(feature_next_plit_2_2,emb)
        feature_next_plit_2 = self.patch_upsample_2(feature_next_plit_2)
        feature_next_plit_2 = self.ConvNeXtBlockAdaLN_8_1(feature_next_plit_2,emb)


        feature_next_plit_3_1,_ = \
            self.compress_3_1.forward_plit(feature=feature_next_plit_2,z_hat=z_hat_3_1,cbr_z=cbr_z_3_1,lmb_embedding=emb,block=1,threshold=4)
        feature_next_plit_3_2,_ = \
            self.compress_3_2.forward_plit(feature=feature_next_plit_3_1,z_hat=z_hat_3_2,cbr_z=cbr_z_3_2,lmb_embedding=emb,block=1,threshold=5)
        feature_next_plit_3_3,_ = \
            self.compress_3_3.forward_plit(feature=feature_next_plit_3_2,z_hat=z_hat_3_3,cbr_z=cbr_z_3_3,lmb_embedding=emb,block=1,threshold=6)
        feature_next_plit_3 = self.ConvNeXtBlockAdaLN_8_2(feature_next_plit_3_3, emb)
        feature_next_plit_3 = self.patch_upsample_3(feature_next_plit_3)
        feature_next_plit_3 = self.ConvNeXtBlockAdaLN_9_1(feature_next_plit_3, emb)

        feature_next_plit_4_1,_ = \
            self.compress_4_1.forward_plit(feature=feature_next_plit_3,z_hat=z_hat_4_1,cbr_z=cbr_z_4_1,lmb_embedding=emb,block=1,threshold=7)
        feature_next_plit_4_2,_ = \
            self.compress_4_2.forward_plit(feature=feature_next_plit_4_1,z_hat=z_hat_4_2,cbr_z=cbr_z_4_2,lmb_embedding=emb,block=1,threshold=8)
        feature_next_plit_4_3,_ = \
            self.compress_4_3.forward_plit(feature=feature_next_plit_4_2,z_hat=z_hat_4_3,cbr_z=cbr_z_4_3,lmb_embedding=emb,block=1,threshold=9)
        feature_next_plit_4 = self.ConvNeXtBlockAdaLN_9_2(feature_next_plit_4_3, emb)
        feature_next_plit_4 = self.patch_upsample_4(feature_next_plit_4)
        feature_next_plit_4 = self.ConvNeXtBlockAdaLN_10_1(feature_next_plit_4, emb)

        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_2(feature_next_plit_4, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_3(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_4(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_5(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_6(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_7(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_8(feature_next_plit_5, emb)
        feature_next_plit_5 = self.patch_upsample_5(feature_next_plit_5)
        feature_next_plit_5_nopad=feature_next_plit_5[:,:,:H,:W]

        mse_plit = tnf.mse_loss(feature_next_plit_5_nopad, x_target, reduction='none').mean(dim=(1, 2, 3))
        out_loss_plit_1 = mse_plit * weight
        x_hat_plit = feature_next_plit_5_nopad
        im_hat_plit = process_output(x_hat_plit.detach())
        im_hat_plit_nopad = im_hat_plit[:,:,:H,:W]
        im_mse_plit = tnf.mse_loss(im_hat_plit_nopad, im, reduction='mean')
        psnr_plit_1 = -10 * math.log10(im_mse_plit.item())


        #node=5
        feature_next_plit_1,_ = \
            self.compress_1.forward_plit(feature=feature_next_0,z_hat=z_hat_1,cbr_z=cbr_z_1,lmb_embedding=emb,block=2,threshold=1)
        feature_next_plit_1 = self.ConvNeXtBlockAdaLN_6_1(feature_next_plit_1,emb)
        feature_next_plit_1 = self.patch_upsample_1(feature_next_plit_1)
        feature_next_plit_1 = self.ConvNeXtBlockAdaLN_7_1(feature_next_plit_1,emb)

        feature_next_plit_2_1,_ = \
            self.compress_2_1.forward_plit(feature=feature_next_plit_1,z_hat=z_hat_2_1,cbr_z=cbr_z_2_1,lmb_embedding=emb,block=2,threshold=2)
        feature_next_plit_2_2,_ = \
            self.compress_2_2.forward_plit(feature=feature_next_plit_2_1,z_hat=z_hat_2_2,cbr_z=cbr_z_2_2,lmb_embedding=emb,block=2,threshold=3)
        feature_next_plit_2 = self.ConvNeXtBlockAdaLN_7_2(feature_next_plit_2_2,emb)
        feature_next_plit_2 = self.patch_upsample_2(feature_next_plit_2)
        feature_next_plit_2 = self.ConvNeXtBlockAdaLN_8_1(feature_next_plit_2,emb)


        feature_next_plit_3_1,_ = \
            self.compress_3_1.forward_plit(feature=feature_next_plit_2,z_hat=z_hat_3_1,cbr_z=cbr_z_3_1,lmb_embedding=emb,block=2,threshold=4)
        feature_next_plit_3_2,_ = \
            self.compress_3_2.forward_plit(feature=feature_next_plit_3_1,z_hat=z_hat_3_2,cbr_z=cbr_z_3_2,lmb_embedding=emb,block=2,threshold=5)
        feature_next_plit_3_3,_ = \
            self.compress_3_3.forward_plit(feature=feature_next_plit_3_2,z_hat=z_hat_3_3,cbr_z=cbr_z_3_3,lmb_embedding=emb,block=2,threshold=6)
        feature_next_plit_3 = self.ConvNeXtBlockAdaLN_8_2(feature_next_plit_3_3, emb)
        feature_next_plit_3 = self.patch_upsample_3(feature_next_plit_3)
        feature_next_plit_3 = self.ConvNeXtBlockAdaLN_9_1(feature_next_plit_3, emb)

        feature_next_plit_4_1,_ = \
            self.compress_4_1.forward_plit(feature=feature_next_plit_3,z_hat=z_hat_4_1,cbr_z=cbr_z_4_1,lmb_embedding=emb,block=2,threshold=7)
        feature_next_plit_4_2,_ = \
            self.compress_4_2.forward_plit(feature=feature_next_plit_4_1,z_hat=z_hat_4_2,cbr_z=cbr_z_4_2,lmb_embedding=emb,block=2,threshold=8)
        feature_next_plit_4_3,_ = \
            self.compress_4_3.forward_plit(feature=feature_next_plit_4_2,z_hat=z_hat_4_3,cbr_z=cbr_z_4_3,lmb_embedding=emb,block=2,threshold=9)
        feature_next_plit_4 = self.ConvNeXtBlockAdaLN_9_2(feature_next_plit_4_3, emb)
        feature_next_plit_4 = self.patch_upsample_4(feature_next_plit_4)
        feature_next_plit_4 = self.ConvNeXtBlockAdaLN_10_1(feature_next_plit_4, emb)

        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_2(feature_next_plit_4, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_3(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_4(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_5(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_6(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_7(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_8(feature_next_plit_5, emb)
        feature_next_plit_5 = self.patch_upsample_5(feature_next_plit_5)
        feature_next_plit_5_nopad=feature_next_plit_5[:,:,:H,:W]

        mse_plit = tnf.mse_loss(feature_next_plit_5_nopad, x_target, reduction='none').mean(dim=(1, 2, 3))
        out_loss_plit_2 = mse_plit * weight
        x_hat_plit = feature_next_plit_5_nopad
        im_hat_plit = process_output(x_hat_plit.detach())
        im_hat_plit_nopad = im_hat_plit[:,:,:H,:W]
        im_mse_plit = tnf.mse_loss(im_hat_plit_nopad, im, reduction='mean')
        psnr_plit_2 = -10 * math.log10(im_mse_plit.item())

        #node=7
        feature_next_plit_1,_ = \
            self.compress_1.forward_plit(feature=feature_next_0,z_hat=z_hat_1,cbr_z=cbr_z_1,lmb_embedding=emb,block=3,threshold=1)
        feature_next_plit_1 = self.ConvNeXtBlockAdaLN_6_1(feature_next_plit_1,emb)
        feature_next_plit_1 = self.patch_upsample_1(feature_next_plit_1)
        feature_next_plit_1 = self.ConvNeXtBlockAdaLN_7_1(feature_next_plit_1,emb)

        feature_next_plit_2_1,_ = \
            self.compress_2_1.forward_plit(feature=feature_next_plit_1,z_hat=z_hat_2_1,cbr_z=cbr_z_2_1,lmb_embedding=emb,block=3,threshold=2)
        feature_next_plit_2_2,_= \
            self.compress_2_2.forward_plit(feature=feature_next_plit_2_1,z_hat=z_hat_2_2,cbr_z=cbr_z_2_2,lmb_embedding=emb,block=3,threshold=3)
        feature_next_plit_2 = self.ConvNeXtBlockAdaLN_7_2(feature_next_plit_2_2,emb)
        feature_next_plit_2 = self.patch_upsample_2(feature_next_plit_2)
        feature_next_plit_2 = self.ConvNeXtBlockAdaLN_8_1(feature_next_plit_2,emb)


        feature_next_plit_3_1,_ = \
            self.compress_3_1.forward_plit(feature=feature_next_plit_2,z_hat=z_hat_3_1,cbr_z=cbr_z_3_1,lmb_embedding=emb,block=3,threshold=4)
        feature_next_plit_3_2,_= \
            self.compress_3_2.forward_plit(feature=feature_next_plit_3_1,z_hat=z_hat_3_2,cbr_z=cbr_z_3_2,lmb_embedding=emb,block=3,threshold=5)
        feature_next_plit_3_3,_ = \
            self.compress_3_3.forward_plit(feature=feature_next_plit_3_2,z_hat=z_hat_3_3,cbr_z=cbr_z_3_3,lmb_embedding=emb,block=3,threshold=6)
        feature_next_plit_3 = self.ConvNeXtBlockAdaLN_8_2(feature_next_plit_3_3, emb)
        feature_next_plit_3 = self.patch_upsample_3(feature_next_plit_3)
        feature_next_plit_3 = self.ConvNeXtBlockAdaLN_9_1(feature_next_plit_3, emb)

        feature_next_plit_4_1,_= \
            self.compress_4_1.forward_plit(feature=feature_next_plit_3,z_hat=z_hat_4_1,cbr_z=cbr_z_4_1,lmb_embedding=emb,block=3,threshold=7)
        feature_next_plit_4_2,_ = \
            self.compress_4_2.forward_plit(feature=feature_next_plit_4_1,z_hat=z_hat_4_2,cbr_z=cbr_z_4_2,lmb_embedding=emb,block=3,threshold=8)
        feature_next_plit_4_3,_= \
            self.compress_4_3.forward_plit(feature=feature_next_plit_4_2,z_hat=z_hat_4_3,cbr_z=cbr_z_4_3,lmb_embedding=emb,block=3,threshold=9)
        feature_next_plit_4 = self.ConvNeXtBlockAdaLN_9_2(feature_next_plit_4_3, emb)
        feature_next_plit_4 = self.patch_upsample_4(feature_next_plit_4)
        feature_next_plit_4 = self.ConvNeXtBlockAdaLN_10_1(feature_next_plit_4, emb)

        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_2(feature_next_plit_4, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_3(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_4(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_5(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_6(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_7(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_8(feature_next_plit_5, emb)
        feature_next_plit_5 = self.patch_upsample_5(feature_next_plit_5)
        feature_next_plit_5_nopad=feature_next_plit_5[:,:,:H,:W]

        mse_plit = tnf.mse_loss(feature_next_plit_5_nopad, x_target, reduction='none').mean(dim=(1, 2, 3))
        out_loss_plit_3 = mse_plit * weight
        x_hat_plit = feature_next_plit_5_nopad
        im_hat_plit = process_output(x_hat_plit.detach())
        im_hat_plit_nopad = im_hat_plit[:,:,:H,:W]
        im_mse_plit = tnf.mse_loss(im_hat_plit_nopad, im, reduction='mean')
        psnr_plit_3 = -10 * math.log10(im_mse_plit.item())

        #node=9
        feature_next_plit_1,_= \
            self.compress_1.forward_plit(feature=feature_next_0,z_hat=z_hat_1,cbr_z=cbr_z_1,lmb_embedding=emb,block=4,threshold=1)
        feature_next_plit_1 = self.ConvNeXtBlockAdaLN_6_1(feature_next_plit_1,emb)
        feature_next_plit_1 = self.patch_upsample_1(feature_next_plit_1)
        feature_next_plit_1 = self.ConvNeXtBlockAdaLN_7_1(feature_next_plit_1,emb)

        feature_next_plit_2_1,_= \
            self.compress_2_1.forward_plit(feature=feature_next_plit_1,z_hat=z_hat_2_1,cbr_z=cbr_z_2_1,lmb_embedding=emb,block=4,threshold=2)
        feature_next_plit_2_2,_ = \
            self.compress_2_2.forward_plit(feature=feature_next_plit_2_1,z_hat=z_hat_2_2,cbr_z=cbr_z_2_2,lmb_embedding=emb,block=4,threshold=3)
        feature_next_plit_2 = self.ConvNeXtBlockAdaLN_7_2(feature_next_plit_2_2,emb)
        feature_next_plit_2 = self.patch_upsample_2(feature_next_plit_2)
        feature_next_plit_2 = self.ConvNeXtBlockAdaLN_8_1(feature_next_plit_2,emb)

        feature_next_plit_3_1,_= \
            self.compress_3_1.forward_plit(feature=feature_next_plit_2,z_hat=z_hat_3_1,cbr_z=cbr_z_3_1,lmb_embedding=emb,block=4,threshold=4)
        feature_next_plit_3_2,_ = \
            self.compress_3_2.forward_plit(feature=feature_next_plit_3_1,z_hat=z_hat_3_2,cbr_z=cbr_z_3_2,lmb_embedding=emb,block=4,threshold=5)
        feature_next_plit_3_3,_ = \
            self.compress_3_3.forward_plit(feature=feature_next_plit_3_2,z_hat=z_hat_3_3,cbr_z=cbr_z_3_3,lmb_embedding=emb,block=4,threshold=6)
        feature_next_plit_3 = self.ConvNeXtBlockAdaLN_8_2(feature_next_plit_3_3, emb)
        feature_next_plit_3 = self.patch_upsample_3(feature_next_plit_3)
        feature_next_plit_3 = self.ConvNeXtBlockAdaLN_9_1(feature_next_plit_3, emb)

        feature_next_plit_4_1,_ = \
            self.compress_4_1.forward_plit(feature=feature_next_plit_3,z_hat=z_hat_4_1,cbr_z=cbr_z_4_1,lmb_embedding=emb,block=4,threshold=7)
        feature_next_plit_4_2,_ = \
            self.compress_4_2.forward_plit(feature=feature_next_plit_4_1,z_hat=z_hat_4_2,cbr_z=cbr_z_4_2,lmb_embedding=emb,block=4,threshold=8)
        feature_next_plit_4_3,_ = \
            self.compress_4_3.forward_plit(feature=feature_next_plit_4_2,z_hat=z_hat_4_3,cbr_z=cbr_z_4_3,lmb_embedding=emb,block=4,threshold=9)
        feature_next_plit_4 = self.ConvNeXtBlockAdaLN_9_2(feature_next_plit_4_3, emb)
        feature_next_plit_4 = self.patch_upsample_4(feature_next_plit_4)
        feature_next_plit_4 = self.ConvNeXtBlockAdaLN_10_1(feature_next_plit_4, emb)

        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_2(feature_next_plit_4, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_3(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_4(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_5(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_6(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_7(feature_next_plit_5, emb)
        feature_next_plit_5 = self.ConvNeXtBlockAdaLN_10_8(feature_next_plit_5, emb)
        feature_next_plit_5 = self.patch_upsample_5(feature_next_plit_5)
        feature_next_plit_5_nopad=feature_next_plit_5[:,:,:H,:W]

        mse_plit = tnf.mse_loss(feature_next_plit_5_nopad, x_target, reduction='none').mean(dim=(1, 2, 3))
        out_loss_plit_4 = mse_plit * weight
        x_hat_plit = feature_next_plit_5_nopad
        im_hat_plit = process_output(x_hat_plit.detach())
        im_hat_plit_nopad = im_hat_plit[:,:,:H,:W]
        im_mse_plit = tnf.mse_loss(im_hat_plit_nopad, im, reduction='mean')
        psnr_plit_4 = -10 * math.log10(im_mse_plit.item())

        # calculate parameter
        out_loss_plit=(out_loss_plit_1+out_loss_plit_2+out_loss_plit_3+out_loss_plit_4)/4
        cbr_z=cbr_z_1+cbr_z_2_1+cbr_z_2_2+cbr_z_3_1+cbr_z_3_2+cbr_z_3_3+cbr_z_4_1+cbr_z_4_2+cbr_z_4_3
        kl_divergences=kl_1.sum(dim=(1,2,3))+kl_2_1.sum(dim=(1,2,3))+kl_2_2.sum(dim=(1,2,3))+kl_3_1.sum(dim=(1,2,3))+kl_3_2.sum(dim=(1,2,3))+\
           kl_3_3.sum(dim=(1,2,3))+kl_4_1.sum(dim=(1,2,3))+kl_4_2.sum(dim=(1,2,3))+kl_4_3.sum(dim=(1,2,3))
        kl = sum(kl_divergences) /B/C/H/W
        bpp=kl*3*math.log2(math.e)
        loss=(out_loss_plit+out_loss_ntc).mean(0)+kl

        return loss,psnr_ntc,psnr_plit_1,psnr_plit_2,psnr_plit_3,psnr_plit_4,kl,cbr_z,bpp

