# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Faster, export-friendly inference graph for DepthPro (plain PyTorch ops only: ONNX / TensorRT exportable).

Same math as DepthPro.forward / DepthPro.infer, reorganised:
- patch embedding once per pyramid level on the whole image (the 384^2 windows start on multiples of 16 px, so every window's
  16x16 patches are patches of the full-image grid), the 35 windows are then gathered as token rows (no image crops / copies);
- merge of the 5x5 / 3x3 window encodings = one row gather with a precomputed index (no slicing + concatenation);
  the final ViT LayerNorm only on the merged tokens (it is per token);
- ConvTranspose2d (2x2, stride 2) followed by a 1x1 conv (decoder fusion blocks 1-4) = one ConvTranspose2d with the
  composed weight and bias (no out_conv pass);
- head: ConvTranspose2d(2x2, s2) + Conv3x3 at 1536^2 = one 3x3 conv at 768^2 producing the 4 output phases
  (composed weights; the deconvolution bias, which the 3x3 conv sees as zero outside the image, enters through a constant
  "ones" channel produced by the preceding conv), the final 1x1 conv per phase, then pixel_shuffle -> 1536^2;
- the FOV network is skipped when the focal length is given (upstream computes it and discards the estimate).

    model, _ = create_model_and_transforms(device=cuda, precision=torch.half)
    fast = DepthProFast(model)                       # shares the weights, builds small composed tensors
    pred = fast.infer_image(img_uint8_hwc_cuda, f_px=None)   # = model.infer(transform(img), f_px)
    canonical_inverse_depth, fov_deg = fast(x)       # = model.forward(x) for x [1, 3, 1536, 1536] (exportable core)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _window_token_index(steps: int, stride_tok: int, grid: int, n: int) -> torch.Tensor:
    """[steps^2, n*n] token indices of each sliding window in a grid x grid token map (same order as DepthProEncoder.split)."""
    out = []
    for j in range(steps):
        for i in range(steps):
            r = torch.arange(n) + j * stride_tok
            c = torch.arange(n) + i * stride_tok
            out.append((r[:, None] * grid + c[None, :]).flatten())
    return torch.stack(out)


def _compose_deconv_1x1(deconv: nn.ConvTranspose2d, conv: nn.Conv2d) -> nn.ConvTranspose2d:
    """conv1x1(deconv(x)) -> one ConvTranspose2d (deconv without bias)."""
    assert deconv.bias is None and conv.kernel_size == (1, 1)
    wd = deconv.weight.float()                                  # [Cin, Cmid, 2, 2]
    wo = conv.weight.float()[:, :, 0, 0]                        # [Cout, Cmid]
    m = nn.ConvTranspose2d(wd.shape[0], wo.shape[0], kernel_size=deconv.kernel_size, stride=deconv.stride, bias=True)
    m.weight.data = torch.einsum("icab,kc->ikab", wd, wo)
    m.bias.data = conv.bias.float().clone()
    return m.to(deconv.weight.dtype)


class _Head(nn.Module):
    """head = Conv3x3(256->128) -> ConvT(2x2,s2) -> Conv3x3(128->32) -> ReLU -> Conv1x1(32->1) -> ReLU, at 768^2 + pixel_shuffle."""

    def __init__(self, head: nn.Sequential):
        super().__init__()
        c0, dc, c2, c4 = head[0], head[1], head[2], head[4]
        cm = c0.out_channels                                     # 128
        ca = cm + 8                                              # + ones channel (+7 zero channels: multiple of 8)
        dt = c0.weight.dtype
        w0 = torch.zeros(ca, *c0.weight.shape[1:])
        w0[:cm] = c0.weight.float().cpu()
        b0 = torch.zeros(ca)
        b0[:cm] = c0.bias.float().cpu()
        b0[cm] = 1.0
        self.conv0 = nn.Conv2d(c0.in_channels, ca, 3, padding=1)
        self.conv0.weight.data, self.conv0.bias.data = w0, b0
        dev = c0.weight.device
        wd, bd = dc.weight.float().cpu(), dc.bias.float().cpu()  # [128, 128, 2, 2], [128]
        w3, b3 = c2.weight.float().cpu(), c2.bias.float().cpu()  # [32, 128, 3, 3], [32]
        k3 = w3.shape[0]
        wc = torch.zeros(k3 * 4, ca, 3, 3)
        for a in range(2):
            for b in range(2):
                for dy in range(3):
                    ty = a + dy - 1
                    py, ay = ty // 2, ty % 2
                    for dx in range(3):
                        tx = b + dx - 1
                        px, ax = tx // 2, tx % 2
                        o = torch.arange(k3) * 4 + a * 2 + b
                        wc[o, :cm, py + 1, px + 1] += w3[:, :, dy, dx] @ wd[:, :, ay, ax].t()
                        wc[o, cm, py + 1, px + 1] += w3[:, :, dy, dx] @ bd
        self.conv1 = nn.Conv2d(ca, k3 * 4, 3, padding=1)
        self.conv1.weight.data, self.conv1.bias.data = wc, b3.repeat_interleave(4)
        w4 = c4.weight.float().cpu()[0, :, 0, 0]                 # [32]
        wz = torch.zeros(4, k3 * 4, 1, 1)
        for p in range(4):
            wz[p, torch.arange(k3) * 4 + p, 0, 0] = w4
        self.conv2 = nn.Conv2d(k3 * 4, 4, 1)
        self.conv2.weight.data, self.conv2.bias.data = wz, c4.bias.float().cpu().repeat(4)
        self.to(dev, dt)

    def forward(self, x):
        y = F.relu(self.conv1(self.conv0(x)))
        return F.pixel_shuffle(F.relu(self.conv2(y)), 2)


class DepthProFast(nn.Module):
    def __init__(self, model):
        super().__init__()
        enc = model.encoder
        self.model = [model]  # not a submodule (weights are shared through the modules below)
        self.pe, self.ie = enc.patch_encoder, enc.image_encoder
        self.hooks = list(enc.hook_block_ids[:2])
        self.up_latent0, self.up_latent1 = enc.upsample_latent0, enc.upsample_latent1
        self.up0, self.up1, self.up2 = enc.upsample0, enc.upsample1, enc.upsample2
        self.up_lowres, self.fuse_lowres = enc.upsample_lowres, enc.fuse_lowres
        dec = model.decoder
        self.dconvs = dec.convs
        self.fusions = dec.fusions
        self.fdeconv = nn.ModuleList([nn.Identity()] + [_compose_deconv_1x1(f.deconv, f.out_conv) for f in dec.fusions[1:]])
        self.head = _Head(model.head)
        self.fov = getattr(model, "fov", None)
        n = enc.out_size                                          # 24 tokens per window side
        self.n = n
        img = enc.img_size                                        # 1536
        ps = self.pe.patch_embed.patch_size[0]                    # 16
        assert self.pe.patch_embed.norm.__class__ is nn.Identity
        g0, g1, g2 = img // ps, img // 2 // ps, img // 4 // ps    # 96, 48, 24
        i0 = _window_token_index(5, int(384 * 0.75) // ps, g0, n)
        i1 = _window_token_index(3, int(384 * 0.5) // ps, g1, n) + g0 * g0
        i2 = _window_token_index(1, 0, g2, n) + g0 * g0 + g1 * g1
        assert 384 * 0.75 % ps == 0 and 384 * 0.5 % ps == 0
        self.register_buffer("tok_idx", torch.cat([i0, i1, i2]), persistent=False)   # [35, 576]
        self.nw = (len(i0), len(i1), len(i2))
        # merge indices: run the upstream merge on a map of row ids (rows of the [35 * (1 + n*n), C] token matrix)
        rows = torch.arange(sum(self.nw) * (n * n + 1)).view(sum(self.nw), 1, n * n + 1)[:, :, 1:].reshape(-1, 1, n, n)
        m0 = enc.merge(rows[: self.nw[0]].float(), batch_size=1, padding=3)[0, 0].long()
        m1 = enc.merge(rows[self.nw[0]: self.nw[0] + self.nw[1]].float(), batch_size=1, padding=6)[0, 0].long()
        m2 = rows[-1, 0].long()
        self.register_buffer("m0", m0, persistent=False)        # [96, 96]
        self.register_buffer("m1", m1, persistent=False)        # [48, 48]
        self.register_buffer("m2", m2, persistent=False)        # [24, 24]
        self.to(model.encoder.patch_encoder.pos_embed.device)

    # ------------------------------------------------------------------------------------------------ network core
    @staticmethod
    def _nchw(t):  # [h, w, C] -> [1, C, h, w] (channels-last view, no copy)
        return t.permute(2, 0, 1).unsqueeze(0)

    def encode(self, x):
        pe = self.pe
        x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        proj = pe.patch_embed.proj
        e = torch.cat([proj(x).flatten(2), proj(x1).flatten(2), proj(x2).flatten(2)], 2)[0].t()   # [96^2+48^2+24^2, C]
        t = e[self.tok_idx]                                                                      # [35, 576, C]
        b = t.shape[0]
        t = torch.cat([pe.cls_token.expand(b, -1, -1), t], 1) + pe.pos_embed
        t = pe.norm_pre(t)
        lat = []
        for i, blk in enumerate(pe.blocks):
            t = blk(t)
            if i in self.hooks:
                lat.append(t)
        C = t.shape[-1]
        T = t.reshape(-1, C)
        f0 = pe.norm(T[self.m0])
        f1 = pe.norm(T[self.m1])
        f2 = pe.norm(T[self.m2])
        l0 = lat[0].reshape(-1, C)[self.m0]
        l1 = lat[1].reshape(-1, C)[self.m0]
        g = self.ie(x2)[0, 1:].reshape(self.n, self.n, C)
        return [self.up_latent0(self._nchw(l0)), self.up_latent1(self._nchw(l1)), self.up0(self._nchw(f0)),
                self.up1(self._nchw(f1)),
                self.fuse_lowres(torch.cat((self.up2(self._nchw(f2)), self.up_lowres(self._nchw(g))), dim=1))], x2

    def decode(self, enc):
        d = self.dconvs
        features = d[-1](enc[-1])
        lowres = features
        f = self.fusions
        features = self.fdeconv[-1](f[-1].resnet2(features))
        for i in range(len(enc) - 2, -1, -1):
            x = features + f[i].resnet1(d[i](enc[i]))
            x = f[i].resnet2(x)
            features = self.fdeconv[i](x) if i else f[0].out_conv(x)
        return features, lowres

    def forward(self, x: torch.Tensor, with_fov: bool = True):
        """x [1, 3, 1536, 1536] normalized -> (canonical inverse depth [1, 1, 1536, 1536], fov_deg [1, 1, 1, 1] or None)."""
        enc, x2 = self.encode(x)
        features, lowres = self.decode(enc)
        cid = self.head(features)
        fov = None
        if with_fov and self.fov is not None:
            fv = self.fov
            y = fv.encoder(x2)[:, 1:].permute(0, 2, 1)
            lr = fv.downsample(lowres)
            fov = fv.head(y.reshape_as(lr) + lr)
        return cid, fov

    # ------------------------------------------------------------------------------------------------ inference
    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """uint8 [H, W, 3] (or [3, H, W]) CUDA tensor -> normalized network input (= transform + infer's resize)."""
        if img.shape[-1] == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)
        x = (img.float() / 255.0 - 0.5) / 0.5
        x = x.to(self.pe.pos_embed.dtype).unsqueeze(0)
        s = self.model[0].img_size
        if x.shape[-2:] != (s, s):
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def postprocess(cid, fov_deg, H, W, f_px=None):
        if f_px is None:
            f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        inverse_depth = cid * (W / f_px)
        f_px = f_px.squeeze()
        if cid.shape[-2:] != (H, W):
            inverse_depth = F.interpolate(inverse_depth, size=(H, W), mode="bilinear", align_corners=False)
        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
        return {"depth": depth.squeeze(), "focallength_px": f_px}

    @torch.no_grad()
    def infer_image(self, img: torch.Tensor, f_px: Optional[torch.Tensor] = None, core=None):
        """= DepthPro.infer(transform(img), f_px); core: optional replacement of self.forward (e.g. a TensorRT engine)."""
        H, W = (img.shape[0], img.shape[1]) if img.shape[-1] == 3 and img.shape[0] != 3 else img.shape[-2:]
        x = self.preprocess(img)
        cid, fov = (core or self.forward)(x, f_px is None)
        return self.postprocess(cid, fov, H, W, f_px)
