import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch import nn, einsum


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    # @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb=None, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, CrossAttention):
                 x = layer(x, context)
            elif isinstance(layer, ResnetBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels, num_groups=16):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-5, affine=True)

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv3d(in_channels,in_channels,kernel_size=3,stride=2,padding=0)

        self.conv1 = nn.Conv3d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.norm1 = Normalize(in_channels)
        self.act1 = nn.LeakyReLU()

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1,0,1)
            # x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = F.pad(x, pad=pad, mode='reflect')
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        x = self.act1(self.norm1(self.conv1(x)))
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels,in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,in_channels, out_channels=None, conv_shortcut=False, dropout=0, temb_channels=512, num_groups=16):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.norm1 = Normalize(out_channels, num_groups)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels,out_channels)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.norm2 = Normalize(out_channels, num_groups)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Sequential(
                    nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                    Normalize(out_channels, num_groups)
                )
            else:
                self.nin_shortcut = nn.Sequential(
                    nn.Conv3d(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
                    Normalize(out_channels, num_groups)
                )
        self.act = nn.SiLU()
    def forward(self, x, temb=None):
        h = self.act(self.norm1(self.conv1(x)))

        if temb is not None:
            # print("t: ", temb.shape)
            h = h + self.temb_proj(self.act(temb))[:,:,None,None,None]

        h = self.dropout(self.act(self.norm2(self.conv2(h))))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return self.act(x+h)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(self,channels,num_heads=1,num_head_channels=-1,use_checkpoint=False,use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = Normalize(channels,num_groups=32)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)

        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        # self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        self.proj_out = nn.Conv1d(channels, channels, 1)

    # def forward(self, x):
    #     return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
    #     #return pt_checkpoint(self._forward, x)  # pytorch

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x)) #[b,3c,w*h*d]
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)  #q,k,v [bs*head, raw_ch/head, whd]
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    # @staticmethod
    # def count_flops(model, _x, y):
    #     return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    # @staticmethod
    # def count_flops(model, _x, y):
    #     return count_flops_attn(model, _x, y)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class CrossAttention(nn.Module):
    def __init__(self, query_dim, style_dim, heads=4, dim_head=64):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(style_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(style_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, style_code):
        # x: [batch, channels, height, width, deep] -> [batch, height*width*deep, channels]
        batch_size, channels, h, w, d = x.shape
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch, hwd, channels]

        # 多头注意力
        q = self.to_q(x)  # [batch, hw, inner_dim]
        # print(style_code.shape)
        k = self.to_k(style_code)  # [batch, context_len, inner_dim]
        v = self.to_v(style_code)  # [batch, context_len, inner_dim]

        q = q.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, hwd, dim_head]
        k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, context_len, dim_head]
        v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, context_len, dim_head]

        # 注意力计算
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [batch, heads, hwd, context_len]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [batch, heads, hwd, dim_head]

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dim_head)  # [batch, hw, inner_dim]
        out = self.to_out(out)  # [batch, hwd, channels]
        out = out.permute(0, 2, 1).view(batch_size, channels, h, w, d)  # [batch, channels, h, w]
        return out

# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, style_dim, heads=4, dim_head=32):
#         super(CrossAttention, self).__init__()
#         inner_dim = dim_head * heads
#         self.dim_head = dim_head
#         self.heads = query_dim // dim_head
#         assert query_dim % dim_head == 0, f"q channels {query_dim} is not divisible by num_head_channels {query_dim}"
#         self.scale = dim_head ** -0.5
#
#         self.to_q = nn.Conv1d(query_dim, query_dim, kernel_size=1, stride=1, bias=False)
#         self.to_k = nn.Conv1d(style_dim, query_dim, kernel_size=1, stride=1, bias=False)
#         self.to_v = nn.Conv1d(style_dim, query_dim, kernel_size=1, stride=1, bias=False)
#         self.to_out = nn.Conv1d(query_dim, query_dim, kernel_size=1, stride=1)
#
#         self.proj_out = nn.Conv1d(query_dim, query_dim, 1)
#
#     def forward(self, x, style_code):
#         # x: [batch, channels, height, width, deep] -> [batch, height*width*deep, channels]
#         # print(x.shape)
#         batch_size, channels, h, w, d = x.shape
#
#         x = x.view(batch_size, channels, -1)  #[batch, channels, hwd]         #
#         q = self.to_q(x).permute(0, 2, 1)  # [batch, hwd, channels]
#
#         # style_code [batch, style_dim=2160, 1 ]
#         k = self.to_k(style_code).permute(0, 2, 1)  # [batch, context_len = 1, query_dim]
#         v = self.to_v(style_code).permute(0, 2, 1)  # [batch, context_len = 1, query_dim]
#
#         q = q.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, hwd, dim_head]
#         k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, context_len, dim_head]
#         v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, context_len, dim_head]
#
#         attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [batch, heads, hwd, context_len]
#         attn = F.softmax(attn, dim=-1)
#         out = torch.matmul(attn, v)  # [batch, heads, hwd, dim_head]
#
#         out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dim_head)  # [batch, hw, inner_dim]
#         # print(out.shape,x.shape)
#         out = self.to_out(out.permute(0, 2, 1))  # [batch, channels, hwd]
#         short_cut = self.proj_out(x)  #[batch, channels, hwd]
#         out = out + short_cut
#         # print(out.shape, short_cut.shape,batch_size, channels, h, w, d )
#         out = out.view(batch_size, channels, h, w, d)  # [batch, channels, h, w]
#         return out
