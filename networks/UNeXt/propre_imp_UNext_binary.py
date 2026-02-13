import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# -----------------------
# Utils / blocks
# -----------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise conv applied on tokens (B, N, C) after reshape to (B, C, H, W)."""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dw(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ShiftMLP(nn.Module):
    """
    Même logique que ton shiftmlp, mais un peu nettoyée.
    x: (B, N, C) avec N=H*W
    """
    def __init__(self, dim, mlp_ratio=1, drop=0.0, shift_size=5, act_layer=nn.GELU):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = DWConv(hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

        self.apply(init_weights)

    def _shift(self, x_img, dim_shift):
        # x_img: (B, C, H, W)
        x_img = F.pad(x_img, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0)
        xs = torch.chunk(x_img, self.shift_size, dim=1)
        # shift -pad .. +pad
        shifted = [torch.roll(xc, s, dims=dim_shift) for xc, s in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(shifted, dim=1)
        x_cat = x_cat[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return x_cat

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_img = x.transpose(1, 2).reshape(B, C, H, W)

        # shift vertical (dim=2)
        x_img = self._shift(x_img, dim_shift=2)
        x = x_img.flatten(2).transpose(1, 2)

        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        # shift horizontal (dim=3) sur le "hidden" => reshape avec hidden en channels
        Bh, Nh, Ch = x.shape
        x_img = x.transpose(1, 2).reshape(Bh, Ch, H, W)
        x_img = self._shift(x_img, dim_shift=3)
        x = x_img.flatten(2).transpose(1, 2)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class ShiftedBlock(nn.Module):
    def __init__(self, dim, drop=0.0, drop_path=0.0, mlp_ratio=1.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mlp = ShiftMLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.apply(init_weights)

    def forward(self, x, H, W):
        return x + self.drop_path(self.mlp(self.norm(x), H, W))


class OverlapPatchEmbed(nn.Module):
    """Conv2d -> tokens + LayerNorm"""
    def __init__(self, img_size, patch_size=3, stride=2, in_chans=128, embed_dim=160):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, N, C)
        x = self.norm(x)
        return x, H, W


# -----------------------
# UNext simplified
# -----------------------
class UNextSimple(nn.Module):
    """
    Un UNext "propre":
    - 3 conv downsample blocks
    - 2 patch-embed + shifted blocks (bottleneck)
    - decode avec upsample + skip
    """
    def __init__(
        self,
        num_classes=1,
        in_chans=3,
        img_size=224,
        base_c=16,              # 16 pour "UNext", 8 pour version small
        embed_dims=(128, 160, 256),
        drop_rate=0.0,
        drop_path_rate=0.0,
        return_logits=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.return_logits = return_logits

        c1 = base_c
        c2 = base_c * 2
        c3 = embed_dims[0]  # 128 par défaut

        # Encoder conv stages
        self.enc1 = ConvBNReLU(in_chans, c1)
        self.enc2 = ConvBNReLU(c1, c2)
        self.enc3 = ConvBNReLU(c2, c3)

        # Tokenized stages
        self.patch3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=c3, embed_dim=embed_dims[1])
        self.patch4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        dpr = torch.linspace(0, drop_path_rate, steps=2).tolist()
        self.block3 = ShiftedBlock(dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], mlp_ratio=1.0)
        self.block4 = ShiftedBlock(dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1], mlp_ratio=1.0)

        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        # Decoder convs
        self.dec4 = ConvBNReLU(embed_dims[2], embed_dims[1])  # up from bottleneck
        self.dblock3 = ShiftedBlock(dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], mlp_ratio=1.0)
        self.dnorm3 = nn.LayerNorm(embed_dims[1])

        self.dec3 = ConvBNReLU(embed_dims[1], embed_dims[0])
        self.dblock2 = ShiftedBlock(dim=embed_dims[0], drop=drop_rate, drop_path=dpr[1], mlp_ratio=1.0)
        self.dnorm2 = nn.LayerNorm(embed_dims[0])

        self.dec2 = ConvBNReLU(embed_dims[0], c2)
        self.dec1 = ConvBNReLU(c2, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

        self.apply(init_weights)

    def forward(self, x):
        B = x.size(0)

        # ---- Encoder ----
        t1 = F.max_pool2d(self.enc1(x), kernel_size=2, stride=2)     # (B, c1, H/2, W/2)
        t2 = F.max_pool2d(self.enc2(t1), 2, 2)                      # (B, c2, H/4, W/4)
        t3 = F.max_pool2d(self.enc3(t2), 2, 2)                      # (B, c3, H/8, W/8)

        # ---- Tokens stage (H/16) ----
        tok3, H3, W3 = self.patch3(t3)                              # embed_dims[1]
        tok3 = self.block3(tok3, H3, W3)
        tok3 = self.norm3(tok3)
        t4 = tok3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()  # skip

        # ---- Bottleneck tokens (H/32) ----
        tok4, H4, W4 = self.patch4(t4)                              # embed_dims[2]
        tok4 = self.block4(tok4, H4, W4)
        tok4 = self.norm4(tok4)
        x = tok4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        # ---- Decoder ----
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec4(x)
        x = x + t4

        # token refine at stage (H/16)
        _, C, H, W = x.shape
        tok = x.flatten(2).transpose(1, 2)
        tok = self.dblock3(tok, H, W)
        tok = self.dnorm3(tok)
        x = tok.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H/8
        x = self.dec3(x)
        x = x + t3

        # token refine at stage (H/8)
        _, C, H, W = x.shape
        tok = x.flatten(2).transpose(1, 2)
        tok = self.dblock2(tok, H, W)
        tok = self.dnorm2(tok)
        x = tok.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H/4
        x = self.dec2(x)
        x = x + t2

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H/2
        x = self.dec1(x)
        x = x + t1

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H
        logits = self.head(x)

        if self.return_logits:
            return logits

        # sinon renvoyer des probas (moins recommandé pour l'entraînement)
        if self.num_classes == 1:
            return torch.sigmoid(logits).squeeze(1)   # (B, H, W)
        else:
            return torch.softmax(logits, dim=1)       # (B, C, H, W)


# -----------------------
# Quick test
# -----------------------
if __name__ == "__main__":
    model = UNextSimple(num_classes=1, in_chans=3, img_size=224, base_c=16, return_logits=False).cuda()
    inp = torch.rand(5, 3, 224, 224).cuda()
    out = model(inp)
    print(out.shape)  # (5, 224, 224) si num_classes=1 et return_logits=False