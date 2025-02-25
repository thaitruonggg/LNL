"""
Modified TNT (Transformer in Transformer) with MCA Cross-Attention
Original TNT by Omid Nejati
MCA Cross-Attention adapted from MCANet
"""
import math
import torch
import torch.nn as nn
from functools import partial
import numbers

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from timm.models.registry import register_model
from einops import rearrange


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


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# MCA Cross-Attention module adapted from MCANet
class MCAAttention(nn.Module):
    """ Multi-Head Cross-Attention adapted from MCANet
    """

    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.project_out = nn.Linear(dim, dim)

        # Directional convolutions for spatial modeling
        self.conv0_1 = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.conv0_2 = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.conv1_1 = nn.Conv1d(dim, dim, 11, padding=5, groups=dim)
        self.conv1_2 = nn.Conv1d(dim, dim, 11, padding=5, groups=dim)
        self.conv2_1 = nn.Conv1d(dim, dim, 21, padding=10, groups=dim)
        self.conv2_2 = nn.Conv1d(dim, dim, 21, padding=10, groups=dim)

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # Normalize the input
        x1 = self.norm1(x)

        # Reshape for 1D convolutions along sequence dimension
        x_seq = x1.transpose(1, 2)  # B, C, N

        # Apply 1D convolutions to model long-range dependencies
        attn_00 = self.conv0_1(x_seq)
        attn_01 = self.conv0_2(x_seq)
        attn_10 = self.conv1_1(x_seq)
        attn_11 = self.conv1_2(x_seq)
        attn_20 = self.conv2_1(x_seq)
        attn_21 = self.conv2_2(x_seq)

        # Combine outputs from different kernel sizes
        out1 = attn_00 + attn_10 + attn_20  # Horizontal direction
        out2 = attn_01 + attn_11 + attn_21  # Vertical direction

        # Process combined outputs
        out1 = out1.transpose(1, 2)  # B, N, C
        out2 = out2.transpose(1, 2)  # B, N, C

        out1 = self.project_out(out1)
        out2 = self.project_out(out2)

        # Reshape for multi-head attention
        k1 = rearrange(out1, 'b n (head c) -> b head n c', head=self.num_heads)
        v1 = rearrange(out1, 'b n (head c) -> b head n c', head=self.num_heads)
        k2 = rearrange(out2, 'b n (head c) -> b head n c', head=self.num_heads)
        v2 = rearrange(out2, 'b n (head c) -> b head n c', head=self.num_heads)

        # Cross-directional queries
        q2 = rearrange(out1, 'b n (head c) -> b head n c', head=self.num_heads)
        q1 = rearrange(out2, 'b n (head c) -> b head n c', head=self.num_heads)

        # Normalize for stable attention
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        # Compute cross-attention
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        # Apply attention to values
        out3 = (attn1 @ v1) + q1
        out4 = (attn2 @ v2) + q2

        # Reshape back to original dimensions
        out3 = rearrange(out3, 'b head n c -> b n (head c)')
        out4 = rearrange(out4, 'b head n c -> b n (head c)')

        # Final projection and dropout
        out = self.project_out(out3) + self.project_out(out4) + x
        out = self.proj_drop(out)

        return out, attn1  # Return attention weights for visualization


class Block(nn.Module):
    """ Modified TNT Block with MCA Cross-Attention
    """

    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer (unchanged, still using standard attention)
        self.norm_in = norm_layer(in_dim)
        self.attn_in = MCAAttention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * 4),
                          out_features=in_dim, act_layer=act_layer, drop=drop)

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)

        # Outer transformer (replaced with MCA Cross-Attention)
        self.norm_out = norm_layer(dim)
        self.attn_out = MCAAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, pixel_embed, patch_embed):
        # Inner transformer with MCA Cross-Attention
        x, inner_weights = self.attn_in(self.norm_in(pixel_embed))
        pixel_embed = pixel_embed + self.drop_path(x)
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))

        # Outer transformer with MCA Cross-Attention
        B, N, C = patch_embed.size()
        patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))
        x, outer_weights = self.attn_out(self.norm_out(patch_embed))
        patch_embed = patch_embed + self.drop_path(x)
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))

        return pixel_embed, patch_embed, outer_weights


class PixelEmbed(nn.Module):
    """ Image to Pixel Embedding (unchanged)
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


class TNT_MCA(nn.Module):
    """ Transformer in Transformer with MCA Cross-Attention
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
def tnt_mca_t_patch16_224(pretrained=False, **kwargs):
    model = TNT_MCA(patch_size=16, embed_dim=192, in_dim=12, depth=12, num_heads=3, in_num_head=3,
                    qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_t_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def tnt_mca_s_patch16_224(pretrained=False, **kwargs):
    model = TNT_MCA(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
                    qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def tnt_mca_b_patch16_224(pretrained=False, **kwargs):
    model = TNT_MCA(patch_size=16, embed_dim=640, in_dim=40, depth=12, num_heads=10, in_num_head=4,
                    qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_b_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model