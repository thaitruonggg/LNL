import math
from functools import partial

#from audioop import bias
from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from mmseg.registry import MODELS
from einops import rearrange
from .utils import resize
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'pixel_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_t_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_s_patch16_224': _cfg(
        url='https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


# Helper functions from your code
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class CrossAttention(nn.Module):
    """Cross-Attention based on your provided code, adapted for TNT sequence input."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., LayerNorm_type='WithBias'):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Normalization from your code
        self.norm1 = LayerNorm(dim, LayerNorm_type)

        # Convolutional layers for key/value generation (from your code)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        # Projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Linear layer to project KV input (for sequence compatibility)
        self.kv_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, q_input, kv_input, h=4, w=4):
        """
        q_input: Query input (e.g., patch_embed) - shape (B, N_q, C)
        kv_input: Key/Value input (e.g., pixel_embed) - shape (B, N_kv, C)
        h, w: Assumed spatial dimensions for reshaping (derived from num_patches or num_pixel)
        """
        B, N_q, C = q_input.shape
        _, N_kv, _ = kv_input.shape

        # Reshape sequence to 2D feature map (assuming N_q = h * w)
        q_input_2d = q_input.transpose(1, 2).reshape(B, C, h, w)  # (B, C, H, W)
        kv_input_proj = self.kv_proj(kv_input)  # Project KV input
        kv_input_2d = kv_input_proj.transpose(1, 2).reshape(B, C, h, w)  # Approximate spatial reshape

        # Normalize query input
        x1 = self.norm1(q_input_2d)

        # Generate attention features (from your code)
        attn_00 = self.conv0_1(x1)
        attn_01 = self.conv0_2(x1)
        attn_10 = self.conv1_1(x1)
        attn_11 = self.conv1_2(x1)
        attn_20 = self.conv2_1(x1)
        attn_21 = self.conv2_2(x1)

        out1 = attn_00 + attn_10 + attn_20  # Horizontal
        out2 = attn_01 + attn_11 + attn_21  # Vertical

        # Project to get initial K and V
        out1 = self.project_out(out1)  # (B, C, H, W)
        out2 = self.project_out(out2)  # (B, C, H, W)

        # Rearrange for multi-head attention (using kv_input for cross-attention)
        k1 = rearrange(kv_input_2d, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(kv_input_2d, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(kv_input_2d, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(kv_input_2d, 'b (head c) h w -> b head w (h c)', head=self.num_heads)

        # Query from q_input
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)

        # Normalize
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        # Cross-attention
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out3 = (attn1 @ v1) + q1

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out4 = (attn2 @ v2) + q2

        # Rearrange back to 2D
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Final output
        out = self.project_out(out3) + self.project_out(out4) + q_input_2d
        out = out.reshape(B, C, -1).transpose(1, 2)  # Back to (B, N_q, C)
        out = self.proj_drop(out)

        weights = (attn1 + attn2) / 2
        return out, weights


class Block(nn.Module):
    """TNT Block with Cross-Attention"""

    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_pixel = num_pixel
        self.spatial_size = int(math.sqrt(num_pixel))  # Assuming square patches (e.g., 4x4)

        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = CrossAttention(
            in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * 4),
                          out_features=in_dim, act_layer=act_layer, drop=drop)

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)

        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = CrossAttention(
            dim, num anfit=heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, pixel_embed, patch_embed):
        # Inner transformer: pixel_embed as query, patch_embed as key/value
        B, N_pixel, _ = pixel_embed.shape
        B, N_patch, _ = patch_embed.shape
        x, _ = self.attn_in(self.norm_in(pixel_embed), patch_embed, h=self.spatial_size, w=self.spatial_size)
        pixel_embed = pixel_embed + self.drop_path(x)
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))

        # Outer transformer: patch_embed as query, pixel_embed as key/value
        patch_size = int(math.sqrt((N_patch - 1)))  # Exclude cls_token
        patch_embed_proj = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N_patch - 1, -1))
        patch_embed_full = torch.cat([patch_embed[:, :1], patch_embed_proj], dim=1)
        x, weights = self.attn_out(self.norm_out(patch_embed_full), pixel_embed, h=patch_size, w=patch_size)
        patch_embed = patch_embed_full + self.drop_path(x)
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        return pixel_embed, patch_embed, weights


class PixelEmbed(nn.Module):
    """ Image to Pixel Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_dim=48, stride=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.num_patches = num_patches
        self.in_dim = in_dim
        new_patch_size = math.ceil(patch_size / stride)
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2d(in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride)
        self.unfold = nn.Unfold(kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x, pixel_pos):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)
        x = self.unfold(x)
        x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size, self.new_patch_size)
        x = x + pixel_pos
        x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
        return x


class TNT(nn.Module):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, in_dim=48, depth=12,
                 num_heads=12, in_num_head=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, first_stride=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pixel_embed = PixelEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, in_dim=in_dim, stride=first_stride)
        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size ** 2

        self.norm1_proj = norm_layer(num_pixel * in_dim)
        self.proj = nn.Linear(num_pixel * in_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pixel_pos = nn.Parameter(torch.zeros(1, in_dim, new_patch_size, new_patch_size))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel, num_heads=num_heads, in_num_head=in_num_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.patch_pos, std=.02)
        trunc_normal_(self.pixel_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_pos', 'pixel_pos', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        attn_weights = []
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)

        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        for blk in self.blocks:
            pixel_embed, patch_embed, weights = blk(pixel_embed, patch_embed)
            attn_weights.append(weights)
        patch_embed = self.norm(patch_embed)
        return patch_embed[:, 0], attn_weights

    def forward(self, x, vis=False):
        x, attn_weights = self.forward_features(x)
        x = self.head(x)
        if vis:
            return x, attn_weights
        else:
            return x


@register_model
def tnt_t_patch16_224(pretrained=False, **kwargs):
    model = TNT(patch_size=16, embed_dim=192, in_dim=12, depth=12, num_heads=3, in_num_head=3,
                qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_t_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def tnt_s_patch16_224(pretrained=False, **kwargs):
    model = TNT(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
                qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def tnt_b_patch16_224(pretrained=False, **kwargs):
    model = TNT(patch_size=16, embed_dim=640, in_dim=40, depth=12, num_heads=10, in_num_head=4,
                qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_b_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model