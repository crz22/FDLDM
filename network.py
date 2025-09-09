import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import functools
from basemodel import ResnetBlock,Downsample,Upsample, Normalize, TimestepEmbedSequential
from basemodel import AttentionBlock,zero_module,timestep_embedding,CrossAttention
from LDM.pretrain.resnet_model import resnet10
from utils import caculate_GRAM

############################################# Autoencoder (base) #######################################

class Encoder(nn.Module):
    def __init__(self, in_channels=1, channels=64, out_channel=4,ch_mult=[1,1,2], num_groups=16,resamp_with_conv=True):
        super(Encoder, self).__init__()
        self.layer_in = nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Sequential(
            ResnetBlock(channels, ch_mult[0]*channels,dropout=0,temb_channels=0, num_groups=num_groups),
            Downsample(in_channels=ch_mult[0]*channels,with_conv=resamp_with_conv)
        )
        self.layer2 = nn.Sequential(
            ResnetBlock(ch_mult[0] * channels, ch_mult[1] * channels, dropout=0, temb_channels=0, num_groups=num_groups),
            Downsample(in_channels=ch_mult[1] * channels, with_conv=resamp_with_conv)
        )
        self.layer3 = nn.Sequential(
            ResnetBlock(ch_mult[1] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0, num_groups=num_groups),
            ResnetBlock(ch_mult[2] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0, num_groups=num_groups)
        )
        # self.norm = Normalize(in_channels=ch_mult[2] * channels, num_groups=num_groups)
        # self.layer_out = nn.Conv3d(in_channels=ch_mult[2] * channels, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        # self.act = nn.SiLU()

    def forward(self, x, temb = None):
        h = self.layer_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        # h = self.layer_out(self.act(self.norm(h)))
        # h = self.layer_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self,in_channels=1, channels=64, out_channel=4, ch_mult=[1,1,2], num_groups=16,resamp_with_conv=True):
        super(Decoder, self).__init__()
        self.layer_in = nn.Sequential(nn.Conv3d(out_channel, channels*ch_mult[2], kernel_size=3, stride=1, padding=1),
                                      Normalize(in_channels=channels*ch_mult[2], num_groups=num_groups),
                                      nn.SiLU())
        self.layer1 = nn.Sequential(
            ResnetBlock(ch_mult[2] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0, num_groups=num_groups)
        )
        self.layer2 = nn.Sequential(
            ResnetBlock(ch_mult[2] * channels, ch_mult[1] * channels, dropout=0, temb_channels=0, num_groups=num_groups),
            Upsample(in_channels=ch_mult[1] * channels,with_conv=resamp_with_conv)
        )
        self.layer3 = nn.Sequential(
            ResnetBlock(ch_mult[1] * channels, ch_mult[0] * channels, dropout=0, temb_channels=0,num_groups=num_groups),
            Upsample(in_channels=ch_mult[0] * channels, with_conv=resamp_with_conv)
        )

        self.norm = Normalize(in_channels=ch_mult[0] * channels, num_groups=num_groups)
        self.layer_out = nn.Conv3d(in_channels=ch_mult[0] * channels, out_channels=in_channels, kernel_size=3, stride=1,padding=1,padding_mode='reflect')
        self.act = nn.SiLU()

    def forward(self, x, temb=None):
        h = self.layer_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        # h = self.layer_out(self.act(self.norm(h)))
        h = self.layer_out(h)
        return h

class Autoencoder(nn.Module):
    def __init__(self,in_channels, out_channel=1):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = Encoder(in_channels=in_channels, out_channel=out_channel)

        self.spade = SPADEGenerator()
        # 解码器
        self.decoder = Decoder(in_channels=in_channels, out_channel=out_channel)

    def forward(self, ref_img, label):
        z = self.encoder(ref_img)
        z = self.spade(z,label)
        x_recon = self.decoder(z)
        # print("z: ", z.shape,"x_recon: ",x_recon.shape)
        return x_recon, z

class KLAutoencoder(nn.Module):
    def __init__(self,in_channels, out_channel=1):
        super(KLAutoencoder, self).__init__()
        # 编码器
        self.encoder = Encoder(in_channels=in_channels, out_channel=out_channel)

        self.enc_mu = nn.Conv3d(out_channel, out_channel, 3, padding=1,padding_mode='reflect')  # [batch, out_channel, 7, 7]
        self.enc_logvar = nn.Conv3d(out_channel, out_channel, 3, padding=1,padding_mode='reflect')  # [batch, out_channel, 7, 7]
        self.spade = SPADEGenerator()
        # 解码器
        self.decoder = Decoder(in_channels=in_channels, out_channel=out_channel)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_forward(self, x, label):
        z = self.encoder(x)
        mu = self.enc_mu(z)
        logvar = self.enc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z = self.spade(z,label)
        return z, mu, logvar

    def normalize_tensor(self, x, eps=1e-8):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)+eps)
        return x / (norm_factor + eps)

    def decode_forward(self, z):
        # z = self.normalize_tensor(z)
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, ref_img, label):
        z, mu, logvar = self.encode_forward(ref_img, label)
        x_recon = self.decode_forward(z)
        # print("z: ", z.shape,"x_recon: ",x_recon.shape)
        return x_recon, z , mu, logvar

class VQLayer(nn.Module):
    def __init__(self, num_embeddings=8192, embedding_dim=4):
        super(VQLayer, self).__init__()
        # self.conv = nn.Conv3d(embedding_dim, num_embeddings, 1)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.embedding_dim = embedding_dim

    def normalize_tensor(self, x, eps=1e-8):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)+eps)
        return x / (norm_factor + eps)

    def forward(self, z):
        # z = self.conv(z)
        # z = self.normalize_tensor(z)
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)
        distances = torch.cdist(z_flat, self.embedding.weight)
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + 0.25 * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3)
        return z_q, loss #, encoding_indices

class VQAutoencoder(nn.Module):
    def __init__(self,in_channels, out_channel=1):
        super(VQAutoencoder, self).__init__()
        # 编码器
        self.encoder = Encoder(in_channels=in_channels, out_channel=out_channel)
        self.VQLayer = VQLayer(embedding_dim=out_channel)
        self.spade = SPADEGenerator()
        # 解码器
        self.decoder = Decoder(in_channels=in_channels, out_channel=out_channel)

    def encode_forward(self, x, label):
        z = self.encoder(x)
        z_q, q_loss = self.VQLayer(z)
        z_q_spade = self.spade(z_q, label)
        return z_q_spade, z_q, q_loss

    def decode_forward(self, z):
        # z = self.normalize_tensor(z)
        # z_q, z = self.VQLayer(z)
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, ref_img, label):
        z_q_spade, z_q, q_loss = self.encode_forward(ref_img, label)
        x_recon = self.decode_forward(z_q_spade)
        # print("z: ", z.shape,"x_recon: ",x_recon.shape)
        return x_recon, z_q, q_loss

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3, norm_type='instance'):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 64
        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, ref_sty, label):
        normalized = self.param_free_norm(ref_sty)

        label = F.interpolate(label, size=ref_sty.size()[2:], mode='nearest')
        actv = self.mlp_shared(label)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        # print(normalized.shape,gamma.shape,beta.shape)
        out = normalized * (1 + gamma) + beta
        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv3d(fin, fmiddle, kernel_size=3, padding=1, padding_mode='reflect'))
        self.conv_1 = spectral_norm(nn.Conv3d(fmiddle, fout, kernel_size=3, padding=1, padding_mode='reflect'))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv3d(fin, fout, kernel_size=1, bias=False))

        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc=1, kernel_size=3, norm_type='instance')
        self.norm_1 = SPADE(fmiddle, label_nc=1, kernel_size=3, norm_type='instance')

    def forward(self, x, y):
        x_s = self.shortcut(x, y)
        dx = self.conv_0(self.actvn(self.norm_0(x, y)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, y)))
        out = x_s + dx
        return out

    def shortcut(self, x, y):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADEGenerator(nn.Module):
    def __init__(self,z_dim=4, nf=128):
        super().__init__()
        self.block = nn.ModuleList([
            SPADEResnetBlock(z_dim, nf),
            SPADEResnetBlock(nf, nf*2),
            SPADEResnetBlock(nf*2, nf),
            SPADEResnetBlock(nf, z_dim),
        ])

    def forward(self, x, y):
        for block in self.block:
            x = block(x, y)
            # print("spade: ",x.max(),x.min())
        # print(x.shape)
        # x = self.normalize_tensor(x, eps=1e-8)
        x = Pixel_norm(x)
        return x

def Pixel_norm(x):
    #x [b,c,w,h,d]
    # x_mean = torch.mean(x, dim=(2,3,4), keepdim=True)
    # print(x_std.shape,x_std)
    x_l = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    # print(x_l.shape)
    x_norm = x/(x_l + 1e-5)
    # print(x_norm)
    return x_norm

class Discriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=32, n_layers=3, use_actnorm=False, out_ch=1):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        # if not use_actnorm:
        #     norm_layer = nn.GroupNorm
        # else:
        #     norm_layer = ActNorm
        norm_layer = nn.GroupNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as GroupNorm3d has affine parameters
            use_bias = norm_layer.func != nn.GroupNorm
        else:
            use_bias = norm_layer != nn.GroupNorm

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]  #0 [16,16,16]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),  #1 [8,8,8] 2[4,4,4]
                norm_layer(num_groups=32, num_channels=ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            # print("discriminator_layer: ", n)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        kw = 3
        sequence += [
            spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(num_groups=32, num_channels=ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            spectral_norm(nn.Conv3d(ndf * nf_mult, out_ch, kernel_size=kw, stride=1, padding=padw))]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # h = input
        # for module in self.main:
        #     h = module(h)
        #     print(h.shape)
        return self.main(input)

class PerceptualEncoder(nn.Module):
    def __init__(self,lamb_cont = 1, lamb_sty = 1):
        super(PerceptualEncoder, self).__init__()
        self.net = resnet10(sample_input_D=32,sample_input_H=32,sample_input_W=32,num_seg_classes=2)
        checkpoint = torch.load('/home/crz/crz_short_cut/Neuron_Generation8/LDM/pretrain/resnet_10_23dataset.pth',weights_only=False)
        new_checkpoint = {}
        print("load pre-train checkpoint")
        for k,v in checkpoint['state_dict'].items():
            new_k = k.replace('module.','')
            # print(new_k)
            new_checkpoint[new_k] = v

        checkpoint['state_dict'] = new_checkpoint
        # for name, param in self.net.named_parameters():
        #     print(f"Layer name: {name}")
        self.net.load_state_dict(checkpoint['state_dict']) #strict=False
        for param in self.parameters():
            param.requires_grad = False

        self.lamb_cont = lamb_cont
        self.lamb_sty = lamb_sty

    def normalize_tensor(self, x, eps=1e-8):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)+eps)
        return x / (norm_factor + eps)

    def caculate_content_loss(self,content_feat1,content_feat2):
        feat_num = len(content_feat1)
        # print("cont feat_num", feat_num)
        content_loss = 0
        for i in range(feat_num):
            # feat1, feat2 = self.normalize_tensor(content_feat1[i]), self.normalize_tensor(content_feat2[i])
            feat1, feat2 = content_feat1[i], content_feat2[i]
            content_loss = content_loss + torch.mean(torch.pow(feat1 - feat2, 2))
        return content_loss/feat_num


    def caculate_style_loss(self,style_feat1,style_feat2):
        feat_num = len(style_feat1)
        # print("sty feat_num",feat_num)
        style_loss = 0
        for i in range(feat_num):
            # feat1, feat2 = self.normalize_tensor(style_feat1[i]), self.normalize_tensor(style_feat2[i])
            feat1, feat2 = style_feat1[i], style_feat2[i]
            feat1_GM, feat2_GM = caculate_GRAM(feat1),caculate_GRAM(feat2)
            # print("feat: ",feat1.shape,feat1.max(),feat1.min(),feat2.max(),feat2.min())
            # print("feat_GM: ",feat1_GM.shape,feat1_GM.max(),feat1_GM.min(),feat2.max(),feat2.min())
            style_loss = style_loss + torch.mean(torch.pow(feat1_GM- feat2_GM, 2))
        return style_loss/feat_num

    def forward(self, input, style, label):
        self.net.eval()
        # print("input: ",input.shape,input.max(),input.min())
        # print("style: ",style.shape,style.max(),style.min())
        # print("label: ",label.shape,label.max(),label.min())
        # input_norm, style_norm, label_norm = self.normalize_tensor(input), self.normalize_tensor(style), self.normalize_tensor(label)  # (self.scaling_layer(input), self.scaling_layer(target))
        _, mid_input_feat = self.net(input)
        _, mid_style_feat = self.net(style)
        _, mid_label_feat = self.net(label)
        # print(outs0.shape, outs1.shape)
        feat_len = len(mid_input_feat)
        # print("feat_len", feat_len)

        style_loss = self.lamb_sty * self.caculate_style_loss(mid_input_feat[:feat_len//2],mid_style_feat[:feat_len//2])
        content_loss = self.lamb_cont * self.caculate_content_loss(mid_input_feat[feat_len//2:], mid_label_feat[feat_len//2:])
        # print("style_loss", style_loss, "content_loss", content_loss)
        per_loss = style_loss + content_loss
        return per_loss

############################################# Autoencoder #######################################

class Content_Encoder(nn.Module):
    def __init__(self,in_channels=1, channels=64, out_channel=4,num_groups=16):
        super(Content_Encoder, self).__init__()
        self.IMG_Encoder = Encoder()
        self.LAB_Encoder = Encoder()
        self.share_layer = nn.Sequential(
            ResnetBlock(128, 64, dropout=0, temb_channels=0, num_groups=num_groups),
            ResnetBlock(64, 64, dropout=0, temb_channels=0, num_groups=num_groups),
            nn.Conv3d(64,out_channel,3,1,1,padding_mode="reflect")
        )

    def forward(self, image, label):
        img_content = self.share_layer(self.IMG_Encoder(image))
        lab_content = self.share_layer(self.LAB_Encoder(label))
        return img_content, lab_content

    def IMG_forward(self, image):
        img_content = self.share_layer(self.IMG_Encoder(image))
        return img_content

    def LAB_forward(self, label):
        lab_content = self.share_layer(self.LAB_Encoder(label))
        return lab_content

class Style_Encoder(nn.Module):
    def __init__(self, in_channels=1, channels=64, out_channel=4,ch_mult=[1,1,2], num_groups=16,resamp_with_conv=True):
        super(Style_Encoder, self).__init__()
        self.layer_in = nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Sequential(
            ResnetBlock(channels, ch_mult[0] * channels, dropout=0, temb_channels=0, num_groups=num_groups),
            Downsample(in_channels=ch_mult[0] * channels, with_conv=resamp_with_conv)
        )
        self.layer2 = nn.Sequential(
            ResnetBlock(ch_mult[0] * channels, ch_mult[1] * channels, dropout=0, temb_channels=0,num_groups=num_groups),
            Downsample(in_channels=ch_mult[1] * channels, with_conv=resamp_with_conv)
        )
        self.layer3 = nn.Sequential(
            ResnetBlock(ch_mult[1] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0,
                        num_groups=num_groups),
            ResnetBlock(ch_mult[2] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0, num_groups=num_groups)
        )
        # self.norm = Normalize(in_channels=ch_mult[2] * channels, num_groups=num_groups)
        # self.layer_out = nn.Conv3d(in_channels=3 * channels, out_channels=out_channel, kernel_size=1, stride=1)
        # self.act = nn.SiLU()
        # self.layer_out = nn.Conv3d(in_channels=ch_mult[2] * channels, out_channels=out_channel, kernel_size=1, stride=1)

        self.layer_out = nn.Conv3d(in_channels=ch_mult[2] * channels, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        h = self.layer_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        return self.layer_out(h)

# class GMStyle_Encoder(nn.Module):
#     def __init__(self, in_channels=1, channels=64, out_channel=4,ch_mult=[1,1,2], num_groups=16,resamp_with_conv=True):
#         super(GMStyle_Encoder, self).__init__()
#         self.out_channel = out_channel
#         self.layer_in = nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1)
#         self.layer1 = nn.Sequential(
#             ResnetBlock(channels, ch_mult[0] * channels, dropout=0, temb_channels=0, num_groups=num_groups),
#             Downsample(in_channels=ch_mult[0] * channels, with_conv=resamp_with_conv)
#         )
#         # GM_feat_num1 = (ch_mult[0] * channels+1)*(ch_mult[0] * channels)//2
#         # self.GM_layer1 = nn.Linear(GM_feat_num1, channels)
#
#         self.layer2 = nn.Sequential(
#             ResnetBlock(ch_mult[0] * channels, ch_mult[1] * channels, dropout=0, temb_channels=0,num_groups=num_groups),
#             Downsample(in_channels=ch_mult[1] * channels, with_conv=resamp_with_conv)
#         )
#         # GM_feat_num2 = (ch_mult[1] * channels + 1) * (ch_mult[1] * channels) // 2
#         # self.GM_layer2 = nn.Linear(GM_feat_num2, channels)
#
#         self.layer3 = nn.Sequential(
#             ResnetBlock(ch_mult[1] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0,num_groups=num_groups),
#             ResnetBlock(ch_mult[2] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0, num_groups=num_groups)
#         )
#         # GM_feat_num3 = (ch_mult[2] * channels + 1) * (ch_mult[2] * channels) // 2
#         # self.GM_layer3 = nn.Linear(GM_feat_num3, channels)
#
#         # self.norm = Normalize(in_channels=ch_mult[2] * channels, num_groups=num_groups)
#         # self.layer_out = nn.Conv3d(in_channels=3 * channels, out_channels=out_channel*8*8*8, kernel_size=1, stride=1)
#         # self.act = nn.SiLU()
#         self.layer_out = nn.Conv3d(in_channels=ch_mult[2] * channels, out_channels=out_channel, kernel_size=1, stride=1)
#
#     def GRAM_transform(self, feat):
#         b, c, w, h, l = feat.shape
#         feature = feat.view(b, c, -1)
#         feature_t = feature.transpose(1, 2)
#         GRAM = torch.bmm(feature, feature_t)
#         GRAM = GRAM / (w * h * l)
#         # print(feature.max(),feature.min())
#         # print(GRAM.shape,GRAM.max(),GRAM.min())
#         tril_index = torch.tril_indices(c, c)
#         GRAM = GRAM[:, tril_index[0], tril_index[1]].view(b, -1)
#
#         norm_factor = torch.sqrt(torch.sum(GRAM ** 2, dim=1, keepdim=True) + 1e-5)
#         GRAM = GRAM / norm_factor
#         # print(GRAM.shape,GRAM.max(),GRAM.min())
#         return GRAM
#
#     def forward(self, x):
#         h = self.layer_in(x)
#
#         h = self.layer1(h)
#         GM1 =  self.GRAM_transform(h)
#         # h_avg1 = torch.mean(h, dim=[2,3,4], keepdim=True)
#         # GM1 = self.GM_layer1(self.GRAM_transform(h)) #[]
#
#         h = self.layer2(h)
#         GM2 = self.GRAM_transform(h)
#         # h_avg2 = torch.mean(h, dim=[2,3,4], keepdim=True)
#         # GM2 = self.GM_layer2(self.GRAM_transform(h))
#
#         h = self.layer3(h)  #[128,8,8,8]
#         # GM3 = self.GRAM_transform(h)
#         # h_avg3 = torch.mean(h, dim=[2,3,4], keepdim=True)
#         # GM3 = self.GM_layer3(self.GRAM_transform(h))
#
#         # GM = torch.cat([GM1, GM2, GM3], dim=1).view(x.shape[0], -1, 1, 1, 1)
#         GM = torch.cat([GM1, GM2], dim=1).view(x.shape[0], -1, 1, 1, 1)
#         # h_avg = torch.cat([h_avg1, h_avg2, h_avg3], dim=1)
#         # print(GM.shape,GM.max(),GM.min())
#         # out = self.layer_out(GM).view(x.shape[0],self.out_channel,*h.shape[2:5]) #[4*8*8*8,1,1,1]
#         out = self.layer_out(h)
#         # print("GM_EN: ",out.shape,out.max(),out.min())
#         return out,GM #, h_avg
class GMStyle_Encoder(nn.Module):
    def __init__(self, in_channels=1, channels=64, out_channel=4,ch_mult=[1,1,2], num_groups=16,resamp_with_conv=True):
        super(GMStyle_Encoder, self).__init__()
        self.out_channel = out_channel
        self.layer_in = nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Sequential(
            ResnetBlock(channels, ch_mult[0] * channels, dropout=0, temb_channels=0, num_groups=num_groups),
            Downsample(in_channels=ch_mult[0] * channels, with_conv=resamp_with_conv)
        )
        # GM_feat_num1 = (ch_mult[0] * channels+1)*(ch_mult[0] * channels)//2
        # self.GM_layer1 = nn.Linear(GM_feat_num1, channels)

        self.layer2 = nn.Sequential(
            ResnetBlock(ch_mult[0] * channels, ch_mult[1] * channels, dropout=0, temb_channels=0,num_groups=num_groups),
            Downsample(in_channels=ch_mult[1] * channels, with_conv=resamp_with_conv)
        )
        # GM_feat_num2 = (ch_mult[1] * channels + 1) * (ch_mult[1] * channels) // 2
        # self.GM_layer2 = nn.Linear(GM_feat_num2, channels)

        self.layer3 = nn.Sequential(
            ResnetBlock(ch_mult[1] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0,num_groups=num_groups),
            ResnetBlock(ch_mult[2] * channels, ch_mult[2] * channels, dropout=0, temb_channels=0, num_groups=num_groups)
        )
        # GM_feat_num3 = (ch_mult[2] * channels + 1) * (ch_mult[2] * channels) // 2
        # self.GM_layer3 = nn.Linear(GM_feat_num3, channels)

        # self.norm = Normalize(in_channels=ch_mult[2] * channels, num_groups=num_groups)
        # self.layer_out = nn.Conv3d(in_channels=3 * channels, out_channels=out_channel*8*8*8, kernel_size=1, stride=1)
        # self.act = nn.SiLU()
        # self.layer_out = nn.Conv3d(in_channels=ch_mult[2] * channels, out_channels=out_channel, kernel_size=1, stride=1)
        self.layer_out = nn.Conv3d(in_channels=ch_mult[2] * channels, out_channels=out_channel, kernel_size=3, stride=1,padding=1)
    def GRAM_transform(self, feat):
        b, c, w, h, l = feat.shape
        feature = feat.view(b, c, -1)
        feature_t = feature.transpose(1, 2)
        GRAM = torch.bmm(feature, feature_t)
        GRAM = GRAM / (w * h * l)
        # print(feature.max(),feature.min())
        # print(GRAM.shape,GRAM.max(),GRAM.min())
        tril_index = torch.tril_indices(c, c)
        GRAM = GRAM[:, tril_index[0], tril_index[1]].view(b, -1)

        norm_factor = torch.sqrt(torch.sum(GRAM ** 2, dim=1, keepdim=True) + 1e-5)
        GRAM = GRAM / norm_factor
        # print(GRAM.shape,GRAM.max(),GRAM.min())
        return GRAM

    def forward(self, x, sty_mode = 'GM'):
        h = self.layer_in(x)

        h = self.layer1(h)
        GM1 =  self.GRAM_transform(h)
        h_avg1 = torch.mean(h, dim=[2,3,4], keepdim=True)
        # GM1 = self.GM_layer1(self.GRAM_transform(h)) #[]

        h = self.layer2(h)
        GM2 = self.GRAM_transform(h)
        h_avg2 = torch.mean(h, dim=[2,3,4], keepdim=True)
        # GM2 = self.GM_layer2(self.GRAM_transform(h))

        h = self.layer3(h)  #[128,8,8,8]
        GM3 = self.GRAM_transform(h)
        h_avg3 = torch.mean(h, dim=[2,3,4], keepdim=True)
        # GM3 = self.GM_layer3(self.GRAM_transform(h))

        GM = torch.cat([GM1, GM2, GM3], dim=1).view(x.shape[0], -1, 1, 1, 1)
        out = self.layer_out(h)
        # print("GM_EN: ",out.shape,out.max(),out.min())
        if sty_mode == 'avg':
            h_avg = torch.cat([h_avg1, h_avg2, h_avg3], dim=1)
            return out, h_avg
        return out, GM  # , h_avg


class Content_Encoder_without_sharelayer(nn.Module):
    def __init__(self,in_channels=1, channels=64, out_channel=4,num_groups=16):
        super(Content_Encoder_without_sharelayer, self).__init__()
        self.IMG_Encoder = Encoder()
        self.LAB_Encoder = Encoder()
        self.outlayer_img = nn.Sequential(
            ResnetBlock(128, 64, dropout=0, temb_channels=0, num_groups=num_groups),
            ResnetBlock(64, 64, dropout=0, temb_channels=0, num_groups=num_groups),
            nn.Conv3d(64,out_channel,3,1,1,padding_mode="reflect")
        )
        self.outlayer_lab = nn.Sequential(
            ResnetBlock(128, 64, dropout=0, temb_channels=0, num_groups=num_groups),
            ResnetBlock(64, 64, dropout=0, temb_channels=0, num_groups=num_groups),
            nn.Conv3d(64,out_channel,3,1,1,padding_mode="reflect")
        )

    def forward(self, image, label):
        img_content = self.outlayer_img(self.IMG_Encoder(image))
        lab_content = self.outlayer_lab(self.LAB_Encoder(label))
        return img_content, lab_content

    def IMG_forward(self, image):
        img_content = self.outlayer_img(self.IMG_Encoder(image))
        return img_content

    def LAB_forward(self, label):
        lab_content = self.outlayer_lab(self.LAB_Encoder(label))
        return lab_content

class DFcws_Autoencoder(nn.Module):
    def __init__(self,in_channels, out_channel=1):
        super(DFcws_Autoencoder, self).__init__()
        # 编码器
        self.sty_encoder =Style_Encoder()
        self.content_encoder = Content_Encoder_without_sharelayer(out_channel=1)

        self.spade = SPADEGenerator()
        # 解码器
        self.decoder = Decoder(in_channels=in_channels, out_channel=out_channel)

    def forward(self, ref_img, label):
        zcont_img, zcont_label = self.content_encoder(ref_img,label)
        zsty_img = self.sty_encoder(ref_img)
        # zsty_img =
        # print(zcont_img.shape,zcont_label.shape,zsty_img.shape)
        zl2i = self.spade(zsty_img,zcont_label)
        zi2i = self.spade(zsty_img,zcont_img)

        x_l2i = self.decoder(zl2i)
        x_i2i = self.decoder(zi2i)
        # print("z: ", z.shape,"x_recon: ",x_recon.shape)
        return x_l2i, x_i2i

    def get_zl2i(self,ref_img, label):
        zcont_label = self.content_encoder.LAB_forward(label)
        zsty_img = self.sty_encoder(ref_img)
        # print(zsty_img.shape,zcont_label.shape)
        zl2i = self.spade(zsty_img,zcont_label)
        return zl2i

    def get_zcont_img(self, ref_image):
        return self.content_encoder.IMG_forward(ref_image)

    def get_zcont_lab(self, label):
        return self.content_encoder.LAB_forward(label)

class DF_Autoencoder(nn.Module):
    def __init__(self,in_channels, out_channel=1):
        super(DF_Autoencoder, self).__init__()
        # 编码器
        self.sty_encoder =Style_Encoder()
        self.content_encoder = Content_Encoder(out_channel=1)

        self.spade = SPADEGenerator()
        # 解码器
        self.decoder = Decoder(in_channels=in_channels, out_channel=out_channel)

    def forward(self, ref_img, label):
        zcont_img, zcont_label = self.content_encoder(ref_img,label)
        zsty_img = self.sty_encoder(ref_img)
        # zsty_img =
        # print(zcont_img.shape,zcont_label.shape,zsty_img.shape)
        zl2i = self.spade(zsty_img,zcont_label)
        zi2i = self.spade(zsty_img,zcont_img)

        x_l2i = self.decoder(zl2i)
        x_i2i = self.decoder(zi2i)
        # print("z: ", z.shape,"x_recon: ",x_recon.shape)
        return x_l2i, x_i2i

    def get_zl2i(self,ref_img, label):
        zcont_label = self.content_encoder.LAB_forward(label)
        zsty_img = self.sty_encoder(ref_img)
        # print(zsty_img.shape,zcont_label.shape)
        zl2i = self.spade(zsty_img,zcont_label)
        return zl2i

    def get_zcont_img(self, ref_image):
        return self.content_encoder.IMG_forward(ref_image)

    def get_zcont_lab(self, label):
        return self.content_encoder.LAB_forward(label)

class DFGM_Autoencoder(nn.Module):
    def __init__(self,in_channels, out_channel=1):
        super(DFGM_Autoencoder, self).__init__()
        # 编码器
        self.sty_encoder =GMStyle_Encoder()
        self.content_encoder = Content_Encoder(out_channel=1)

        self.spade = SPADEGenerator()
        # 解码器
        self.decoder = Decoder(in_channels=in_channels, out_channel=out_channel)

    def forward(self, ref_img, label):
        zcont_img, zcont_label = self.content_encoder(ref_img,label)
        zsty_img, sty_GM = self.sty_encoder(ref_img)
        # print(zcont_img.shape,zcont_label.shape,zsty_img.shape)
        zl2i = self.spade(zsty_img,zcont_label)
        zi2i = self.spade(zsty_img,zcont_img)

        x_l2i = self.decoder(zl2i)
        x_i2i = self.decoder(zi2i)
        # print("z: ", z.shape,"x_recon: ",x_recon.shape)
        return x_l2i, x_i2i

    def get_zl2i(self,ref_img, label):
        zcont_label = self.content_encoder.LAB_forward(label)
        zsty_img, sty_GM = self.sty_encoder(ref_img)
        # print("get_zl2i",zsty_img.max(),zcont_label.max())
        zl2i = self.spade(zsty_img,zcont_label)
        return zl2i, sty_GM

    def get_zcont_img(self, ref_image):
        return self.content_encoder.IMG_forward(ref_image)

    def get_zcont_lab(self, label):
        return self.content_encoder.LAB_forward(label)

    def get_zsty_img(self, ref_image):
        zsty_img, sty_GM = self.sty_encoder(ref_image)
        return zsty_img, sty_GM

    def get_zl2i_avg(self,ref_img, label):
        zcont_label = self.content_encoder.LAB_forward(label)
        zsty_img, h_avg = self.sty_encoder(ref_img,sty_mode = 'avg')
        # print("get_zl2i",zsty_img.max(),zcont_label.max())
        zl2i = self.spade(zsty_img,zcont_label)
        return zl2i, h_avg

    # def get_zl2i2(self, ref_img, label):
    #     zcont_label = self.content_encoder.LAB_forward(label)
    #     zsty_img, _= self.sty_encoder(ref_img)
    #     # print("get_zl2i",zsty_img.max(),zcont_label.max())
    #     zl2i = self.spade(zsty_img, zcont_label)
    #     return zl2i
class DFVQ_Autoencoder(nn.Module):
    def __init__(self,in_channels, out_channel=1):
        super(DFVQ_Autoencoder, self).__init__()
        # 编码器
        self.sty_encoder =Style_Encoder()
        self.content_encoder = Content_Encoder(out_channel=1)
        self.VQLayer = VQLayer(embedding_dim=out_channel)
        self.spade = SPADEGenerator()
        # 解码器
        self.decoder = Decoder(in_channels=in_channels, out_channel=out_channel)

    def forward(self, ref_img, label):
        zcont_img, zcont_label = self.content_encoder(ref_img,label)
        zsty_img = self.sty_encoder(ref_img)
        # zsty_img =
        # print(zcont_img.shape,zcont_label.shape,zsty_img.shape)
        zsty_img_q, q_loss = self.VQLayer(zsty_img)

        zl2i = self.spade(zsty_img_q,zcont_label)
        zi2i = self.spade(zsty_img_q,zcont_img)

        x_l2i = self.decoder(zl2i)
        x_i2i = self.decoder(zi2i)
        # print("z: ", z.shape,"x_recon: ",x_recon.shape)
        return x_l2i, x_i2i, q_loss

    def get_zl2i(self,ref_img, label):
        zcont_label = self.content_encoder.LAB_forward(label)
        zsty_img = self.sty_encoder(ref_img)
        # print(zsty_img.shape,zcont_label.shape)
        zsty_img_q, _ = self.VQLayer(zsty_img)
        zl2i_q = self.spade(zsty_img_q,zcont_label)
        return zl2i_q

    def get_zcont_img(self, ref_image):
        return self.content_encoder.IMG_forward(ref_image)

    def get_zcont_lab(self, label):
        return self.content_encoder.LAB_forward(label)

class DFVQGM_Autoencoder(nn.Module):
    def __init__(self,in_channels, out_channel=1):
        super(DFVQGM_Autoencoder, self).__init__()
        # 编码器
        self.sty_encoder = GMStyle_Encoder()
        self.content_encoder = Content_Encoder(out_channel=1)
        self.VQLayer = VQLayer(embedding_dim=out_channel)
        self.spade = SPADEGenerator()
        # 解码器
        self.decoder = Decoder(in_channels=in_channels, out_channel=out_channel)

    def forward(self, ref_img, label):
        zcont_img, zcont_label = self.content_encoder(ref_img,label)
        zsty_img, sty_GM = self.sty_encoder(ref_img)

        zsty_img_q, q_loss = self.VQLayer(zsty_img)

        zl2i = self.spade(zsty_img_q,zcont_label)
        zi2i = self.spade(zsty_img_q,zcont_img)

        x_l2i = self.decoder(zl2i)
        x_i2i = self.decoder(zi2i)
        # print("z: ", z.shape,"x_recon: ",x_recon.shape)
        return x_l2i, x_i2i, q_loss

    def get_zl2i(self,ref_img, label):
        zcont_label = self.content_encoder.LAB_forward(label)
        zsty_img, sty_GM = self.sty_encoder(ref_img)
        zsty_img_q, _ = self.VQLayer(zsty_img)
        zl2i_q = self.spade(zsty_img_q,zcont_label)
        return zl2i_q, sty_GM

    def get_zcont_img(self, ref_image):
        return self.content_encoder.IMG_forward(ref_image)

    def get_zcont_lab(self, label):
        return self.content_encoder.LAB_forward(label)

    def get_zsty_img(self, ref_image):
        zsty_img, sty_GM = self.sty_encoder(ref_image)
        return zsty_img, sty_GM

############################################# DM #######################################
class Unet_3D(nn.Module):
    def __init__(self,in_channels, out_channel=1, model_channels=128, dropout = 0,
                 channel_mult=[1,1,2], num_res_blocks=1,
                 attention_resolutions=[4, 2], num_head_channels=32, num_heads=-1, legacy=True,
                 resblock_updown = False):
        super(Unet_3D, self).__init__()
        self.model_channels = model_channels
        num_heads_upsample = num_heads
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        ########################################## Downsample part ##################################################
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, kernel_size=3, stride=1, padding=1,padding_mode='reflect'),),]
        )
        ds = 1
        ch = model_channels
        input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            #ADD resblock and Attblock
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(in_channels = ch,out_channels=mult * model_channels, temb_channels=time_embed_dim, dropout=dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    # if legacy:
                    #     # num_heads = 1
                    #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(ch,num_heads=num_heads,num_head_channels=dim_head)
                        # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))  #[RES+ATT]

                # self._feature_size += ch
                input_block_chans.append(ch)
            #ADD Downsample layer
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResnetBlock(ch,out_channels=out_ch, temb_channels= time_embed_dim, dropout= dropout, down = True)
                        if resblock_updown #default False
                        else Downsample(ch, True)
                    )
                )
                # ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                # self._feature_size += ch
        # print('input_block_chans: ',input_block_chans)
        ########################################## Middle part ##################################################
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        # if legacy:
        #     #num_heads = 1
        #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResnetBlock(ch,out_channels=ch,temb_channels= time_embed_dim, dropout= dropout),
            AttentionBlock(ch,num_heads=num_heads,num_head_channels=dim_head),
                # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResnetBlock(ch,out_channels=ch,temb_channels= time_embed_dim, dropout= dropout),
        )
        # self._feature_size += ch
        ########################################## Upsample part ##################################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # ADD resblock and Attblock
                ich = input_block_chans.pop()
                layers = [ResnetBlock(ch + ich,out_channels=model_channels * mult, temb_channels= time_embed_dim, dropout= dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    # if legacy:
                    #     # num_heads = 1
                    #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(ch,num_heads=num_heads_upsample,num_head_channels=dim_head)
                        # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                # ADD UPsample layer
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResnetBlock(ch,out_channels=out_ch,temb_channels=time_embed_dim, dropout=dropout, up=True)
                        if resblock_updown #default False
                        else Upsample(ch, True)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                # self._feature_size += ch
        ########################################## Output part ##################################################
        self.out = nn.Sequential(
            Normalize(ch,num_groups=32),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channel, 3, padding=1, padding_mode='reflect'),
            # zero_module(nn.Conv3d(model_channels, out_channel, 3, padding=1,padding_mode='reflect')),
        )
        # if self.predict_codebook_ids:
        #     self.id_predictor = nn.Sequential(
        #         normalization(ch),
        #         conv_nd(dims, model_channels, n_embed, 1),
        #         # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        #     )

    def forward(self,x,timesteps=None,context=None):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        # add condition information
        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     emb = emb + self.label_emb(y)
        hs = []
        h = x
        for module in self.input_blocks:
            # print(h.shape)
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = (torch.cat([h, hs.pop()], dim=1))
            h = module(h, emb, context)
        h = h.type(x.dtype)
        # if self.predict_codebook_ids:
        #     return self.id_predictor(h)
        # else:
        #     return self.out(h)
        return self.out(h)

class Unet_3D_CA(nn.Module):
    def __init__(self,in_channels, out_channel=1, model_channels=128, dropout = 0,
                 channel_mult=[1,1,2], num_res_blocks=1,
                 attention_resolutions=[4, 2], num_head_channels=32, num_heads=-1, legacy=True,
                 resblock_updown = False):
        super(Unet_3D_CA, self).__init__()
        self.model_channels = model_channels
        self.style_dim = 4160 #12416
        num_heads_upsample = num_heads
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        ########################################## Downsample part ##################################################
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, kernel_size=3, stride=1, padding=1,padding_mode='reflect'),),]
        )
        ds = 1
        ch = model_channels
        input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            #ADD resblock and Attblock
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(in_channels = ch,out_channels=mult * model_channels, temb_channels=time_embed_dim, dropout=dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    # if legacy:
                    #     # num_heads = 1
                    #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        CrossAttention(query_dim=ch, style_dim=128)
                        # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))  #[RES+ATT]

                # self._feature_size += ch
                input_block_chans.append(ch)
            #ADD Downsample layer
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResnetBlock(ch,out_channels=out_ch, temb_channels= time_embed_dim, dropout= dropout, down = True)
                        if resblock_updown #default False
                        else Downsample(ch, True)
                    )
                )
                # ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                # self._feature_size += ch
        # print('input_block_chans: ',input_block_chans)
        ########################################## Middle part ##################################################
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        # if legacy:
        #     #num_heads = 1
        #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResnetBlock(ch,out_channels=ch,temb_channels= time_embed_dim, dropout= dropout),
            CrossAttention(query_dim=ch, style_dim=128),
                # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResnetBlock(ch,out_channels=ch,temb_channels= time_embed_dim, dropout= dropout),
        )
        # self._feature_size += ch
        ########################################## Upsample part ##################################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # ADD resblock and Attblock
                ich = input_block_chans.pop()
                layers = [ResnetBlock(ch + ich,out_channels=model_channels * mult, temb_channels= time_embed_dim, dropout= dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    # if legacy:
                    #     # num_heads = 1
                    #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        CrossAttention(query_dim=ch, style_dim=128)
                        # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                # ADD UPsample layer
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResnetBlock(ch,out_channels=out_ch,temb_channels=time_embed_dim, dropout=dropout, up=True)
                        if resblock_updown #default False
                        else Upsample(ch, True)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                # self._feature_size += ch
        ########################################## Output part ##################################################
        self.out = nn.Sequential(
            Normalize(ch,num_groups=32),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channel, 3, padding=1, padding_mode='reflect')
            # zero_module(nn.Conv3d(model_channels, out_channel, 3, padding=1,padding_mode='reflect')),
        )

        # self.style_layer = nn.Conv1d(4160, 128, kernel_size=1, stride=1)
        self.style_layer = nn.Conv1d(self.style_dim, 128, kernel_size=1, stride=1)
        # self.style_layer = nn.Linear(12416, 128)
    def forward(self,x,timesteps=None,context=None):
        # context : style_GM code (12416)
        batch_size = x.shape[0]
        # print(context.shape)
        style_code0 = context[:,:self.style_dim]
        style_code = self.style_layer(style_code0.view(batch_size, -1, 1))
        # style_code = self.style_layer(context.view(batch_size, -1))
        style_code = style_code.view(batch_size, 1, -1)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks:
            # print(h.shape)
            h = module(h, emb, style_code)
            hs.append(h)
        # print("h1: ",h.max())
        h = self.middle_block(h, emb, style_code)

        for module in self.output_blocks:
            h = (torch.cat([h, hs.pop()], dim=1))
            h = module(h, emb, style_code)
        h = h.type(x.dtype)

        return self.out(h)

class Unet_3D_CA2(nn.Module):
    def __init__(self,in_channels, out_channel=1, model_channels=128, dropout = 0,
                 channel_mult=[1,1,2], num_res_blocks=1,
                 attention_resolutions=[4, 2], num_head_channels=32, num_heads=-1, legacy=True,
                 resblock_updown = False):
        super(Unet_3D_CA2, self).__init__()
        self.model_channels = model_channels
        num_heads_upsample = num_heads
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        ########################################## Downsample part ##################################################
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, kernel_size=3, stride=1, padding=1,padding_mode='reflect'),),]
        )
        ds = 1
        ch = model_channels
        input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            #ADD resblock and Attblock
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(in_channels = ch,out_channels=mult * model_channels, temb_channels=time_embed_dim, dropout=dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    # if legacy:
                    #     # num_heads = 1
                    #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        CrossAttention(query_dim=ch, style_dim=128)
                        # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))  #[RES+ATT]

                # self._feature_size += ch
                input_block_chans.append(ch)
            #ADD Downsample layer
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResnetBlock(ch,out_channels=out_ch, temb_channels= time_embed_dim, dropout= dropout, down = True)
                        if resblock_updown #default False
                        else Downsample(ch, True)
                    )
                )
                # ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                # self._feature_size += ch
        # print('input_block_chans: ',input_block_chans)
        ########################################## Middle part ##################################################
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        # if legacy:
        #     #num_heads = 1
        #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResnetBlock(ch,out_channels=ch,temb_channels= time_embed_dim, dropout= dropout),
            CrossAttention(query_dim=ch, style_dim=128),
                # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResnetBlock(ch,out_channels=ch,temb_channels= time_embed_dim, dropout= dropout),
        )
        # self._feature_size += ch
        ########################################## Upsample part ##################################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # ADD resblock and Attblock
                ich = input_block_chans.pop()
                layers = [ResnetBlock(ch + ich,out_channels=model_channels * mult, temb_channels= time_embed_dim, dropout= dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    # if legacy:
                    #     # num_heads = 1
                    #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        CrossAttention(query_dim=ch, style_dim=128)
                        # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                # ADD UPsample layer
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResnetBlock(ch,out_channels=out_ch,temb_channels=time_embed_dim, dropout=dropout, up=True)
                        if resblock_updown #default False
                        else Upsample(ch, True)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                # self._feature_size += ch
        ########################################## Output part ##################################################
        self.out = nn.Sequential(
            Normalize(ch,num_groups=32),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channel, 3, padding=1, padding_mode='reflect')
            # zero_module(nn.Conv3d(model_channels, out_channel, 3, padding=1,padding_mode='reflect')),
        )
        self.style_layer = nn.Conv3d(256, 128, kernel_size=1, stride=1)
        # self.style_layer = nn.Linear(12416, 128)
    def forward(self,x,timesteps=None,context=None):
        batch_size = x.shape[0]
        # print(context.shape)
        style_code = self.style_layer(context)
        style_code = style_code.view(batch_size, 1, -1)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks:
            # print(h.shape)
            h = module(h, emb, style_code)
            hs.append(h)
        # print("h1: ",h.max())
        h = self.middle_block(h, emb, style_code)

        for module in self.output_blocks:
            h = (torch.cat([h, hs.pop()], dim=1))
            h = module(h, emb, style_code)
        h = h.type(x.dtype)
        return self.out(h)

class Unet_3D_CA_1(nn.Module):
    def __init__(self,in_channels, out_channel=1, model_channels=128, dropout = 0,
                 channel_mult=[1,1,2], num_res_blocks=1,
                 attention_resolutions=[4, 2], num_head_channels=32, num_heads=-1, legacy=True,
                 resblock_updown = False):
        super(Unet_3D_CA_1, self).__init__()
        self.model_channels = model_channels
        self.style_dim = 4160
        num_heads_upsample = num_heads
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        ########################################## Downsample part ##################################################
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, kernel_size=3, stride=1, padding=1,padding_mode='reflect'),),]
        )
        ds = 1
        ch = model_channels
        input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            #ADD resblock and Attblock
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(in_channels = ch,out_channels=mult * model_channels, temb_channels=time_embed_dim, dropout=dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    # if legacy:
                    #     # num_heads = 1
                    #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        CrossAttention(query_dim=ch, style_dim=self.style_dim)
                        # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))  #[RES+ATT]

                # self._feature_size += ch
                input_block_chans.append(ch)
            #ADD Downsample layer
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResnetBlock(ch,out_channels=out_ch, temb_channels= time_embed_dim, dropout= dropout, down = True)
                        if resblock_updown #default False
                        else Downsample(ch, True)
                    )
                )
                # ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                # self._feature_size += ch
        # print('input_block_chans: ',input_block_chans)
        ########################################## Middle part ##################################################
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        # if legacy:
        #     #num_heads = 1
        #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResnetBlock(ch,out_channels=ch,temb_channels= time_embed_dim, dropout= dropout),
            CrossAttention(query_dim=ch, style_dim=self.style_dim),
                # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResnetBlock(ch,out_channels=ch,temb_channels= time_embed_dim, dropout= dropout),
        )
        # self._feature_size += ch
        ########################################## Upsample part ##################################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # ADD resblock and Attblock
                ich = input_block_chans.pop()
                layers = [ResnetBlock(ch + ich,out_channels=model_channels * mult, temb_channels= time_embed_dim, dropout= dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    # if legacy:
                    #     # num_heads = 1
                    #     dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        CrossAttention(query_dim=ch, style_dim=self.style_dim)
                        # if not use_spatial_transformer else SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                # ADD UPsample layer
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResnetBlock(ch,out_channels=out_ch,temb_channels=time_embed_dim, dropout=dropout, up=True)
                        if resblock_updown #default False
                        else Upsample(ch, True)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                # self._feature_size += ch
        ########################################## Output part ##################################################
        self.out = nn.Sequential(
            Normalize(ch,num_groups=32),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channel, 3, padding=1, padding_mode='reflect')
            # zero_module(nn.Conv3d(model_channels, out_channel, 3, padding=1,padding_mode='reflect')),
        )
        # if self.predict_codebook_ids:
        #     self.id_predictor = nn.Sequential(
        #         normalization(ch),
        #         conv_nd(dims, model_channels, n_embed, 1),
        #         # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        #     )

        # self.style_layer = nn.Conv1d(12416, 128, kernel_size=1, stride=1)
        # self.style_layer = nn.Linear(12416, 128)
    def forward(self,x,timesteps=None,style=None):
        # context : style_GM code (12416)
        batch_size = x.shape[0]
        # print(context.shape)
        # style_code = self.style_layer(context.view(batch_size, -1, 1))
        # style_code = self.style_layer(context.view(batch_size, -1))
        style_ = style[:,:self.style_dim,:,:,:]
        style_code = style_.view(batch_size, self.style_dim, -1) #[4160,1]

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks:
            # print(h.shape)
            h = module(h, emb, style_code)
            hs.append(h)
        # print("h1: ",h.max())
        h = self.middle_block(h, emb, style_code)

        for module in self.output_blocks:
            h = (torch.cat([h, hs.pop()], dim=1))
            h = module(h, emb, style_code)
        h = h.type(x.dtype)
        # if self.predict_codebook_ids:
        #     return self.id_predictor(h)
        # else:
        #     return self.out(h)
        # print(h.shape)
        # print("h2: ", h.max(),h.min(),self.out[0](h).max(),self.out[1](h).max(),self.out[2](h).max())#
        return self.out(h)
