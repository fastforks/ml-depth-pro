"""Depth Pro cuda-kernels path: the portable graph (depth_pro.fast_inference.DepthProFast: token
gathers instead of crops, merge by index, composed deconv/1x1 and head weights, FOV only when needed) with prebuilt kernels:

- ViT-L blocks (patch encoder on 35 x 577 tokens, image encoder, FOV encoder): LayerNorm (torch) -> qkv (CUTLASS, bias epilogue) ->
  fastbb flash attention "hh" (fp16-accumulate QK^T and PV, reads the packed qkv, writes [B, N, C]) -> proj with bias +
  LayerScale + residual in the CUTLASS epilogue -> LayerNorm -> fc1 + GELU epilogue -> fc2 + bias + LayerScale + residual epilogue
  (csrc/dp_fused.cu, fp16 accumulation). No elementwise passes left in the blocks.
- encoder upsampling: 1x1 projection + each 2x2/s2 deconvolution as ONE GEMM over the input pixels whose epilogue writes the 4
  output phases straight into the spatial NHWC map (no pixel-shuffle copies);
- decoder: all 3x3 convs CUTLASS with bias / ReLU / residual fused; the pre-activation ReLUs of the residual blocks come from a
  second epilogue output of the producing conv (y = v + res and y2 = relu(v) or relu(v + res)), so no ReLU / add passes;
  deconv + 1x1 of the fusion blocks composed (phase GEMM with bias); the 768^2 latent0 map is written by its last deconvolution
  together with the decoder's skip sum and its ReLU;
- head at 768^2 (composed deconv + 3x3, ones channel for the deconv bias), last 1x1 per phase, pixel shuffle.
General: Depth Pro always runs at 1536^2 (input resized like upstream infer), so there is no per-size work at all; no CUDA graphs.
"""
import contextlib

import torch
import torch.nn.functional as F

from .dpfused import conv as dconv, load as _load_ext
from .fastbb_vendored import attention as _fa  # prebuilt flash kernel (fastforks/fastbb)


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


def _cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def _w1x1(w):  # Linear [N, K] or Conv1x1 [N, K, 1, 1] -> [N, K, 1, 1] channels-last fp16
    return _cl(w.detach().reshape(w.shape[0], -1, 1, 1).half())


def _phase_w(wd):  # ConvTranspose2d weight [Cin, Cout, 2, 2] -> GEMM weight [4 * Cout, Cin, 1, 1], row = (a * 2 + b) * Cout + c
    ci, co = wd.shape[:2]
    return _cl(wd.detach().permute(2, 3, 1, 0).reshape(4 * co, ci, 1, 1).half())


def _tok4(t):  # [M, C] contiguous -> [1, C, 1, M] channels-last view (the GEMM-as-1x1-conv layout)
    M, C = t.shape
    return t.view(1, 1, M, C).permute(0, 3, 1, 2)


def _rows(x4):  # [1, C, 1, M] channels-last -> [M, C] view
    return x4.permute(0, 2, 3, 1).reshape(-1, x4.shape[1])


class _Blk:
    def __init__(self, b):
        a = b.attn
        self.n1, self.n2 = b.norm1, b.norm2
        self.H = a.num_heads
        self.scale = a.scale
        self.wqkv, self.bqkv = _w1x1(a.qkv.weight), a.qkv.bias
        self.wp, self.bp = _w1x1(a.proj.weight), a.proj.bias
        self.g1 = b.ls1.gamma.detach().half().contiguous()
        self.w1, self.b1 = _w1x1(b.mlp.fc1.weight), b.mlp.fc1.bias
        self.w2, self.b2 = _w1x1(b.mlp.fc2.weight), b.mlp.fc2.bias
        self.g2 = b.ls2.gamma.detach().half().contiguous()
        assert type(b.mlp.act).__name__ == "GELU" and getattr(b.mlp.act, "approximate", "none") == "none"


class _ViT:
    """timm ViT (Depth Pro's DINOv2-L/16 at 384^2) on token rows; blocks through the fused kernels."""
    def __init__(self, vit, attn="hh"):
        self.vit, self.attn = vit, attn
        self.blocks = [_Blk(b) for b in vit.blocks]
        for b in vit.blocks:
            for n in ("q_norm", "k_norm", "norm"):
                assert type(getattr(b.attn, n, torch.nn.Identity())).__name__ == "Identity"

    def block(self, b, x4, B, N):
        M, C = B * N, x4.shape[1]
        y = b.n1(_rows(x4))
        qkv = _rows(dconv(_tok4(y), b.wqkv, b.bqkv)).view(B, N, 3, b.H, C // b.H)
        q, k, v = qkv.unbind(2)
        o = _fa.attention(q, k, v, scale=b.scale, variant=self.attn).view(M, C)
        x4 = dconv(_tok4(o), b.wp, b.bp, scale=b.g1, res=x4)
        h = dconv(_tok4(b.n2(_rows(x4))), b.w1, b.b1, act="gelu")
        return dconv(h, b.w2, b.b2, scale=b.g2, res=x4)

    def tokens_image(self, x):
        """= timm forward_features prologue for one 384^2 image -> [1, 577, C]"""
        v = self.vit
        t = v.patch_embed.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([v.cls_token.expand(t.shape[0], -1, -1), t], 1) + v.pos_embed

    def run(self, t, hooks=()):
        """t [B, N, C] tokens -> (last block output rows [B*N, C] (not normed), {hook i: rows})"""
        B, N, C = t.shape
        x4 = _tok4(t.reshape(B * N, C).contiguous())
        out = {}
        with fp16_accumulate(True):
            for i, b in enumerate(self.blocks):
                x4 = self.block(b, x4, B, N)
                if i in hooks:
                    out[i] = _rows(x4)
        return _rows(x4), out


class DepthProMax:
    def __init__(self, fast, attn="hh", streams=True):
        """fast: depth_pro.fast_inference.DepthProFast (fp16 model). streams: run the two batch-1 ViTs (image encoder, FOV
        encoder; ~40-CTA GEMMs that fill the GPU poorly) on side streams, concurrently with the 35-patch encoder."""
        _load_ext()
        self.streams = [torch.cuda.Stream(), torch.cuda.Stream()] if streams else None
        self.f = f = fast
        self.pvit = _ViT(f.pe, attn)
        self.ivit = _ViT(f.ie, attn)
        self.fvit = _ViT(f.fov.encoder[0], attn) if f.fov is not None else None
        up = lambda s: (_w1x1(s[0].weight), [_phase_w(m.weight) for m in s[1:]])  # noqa: E731
        self.up_l0, self.up_l1, self.up0, self.up1, self.up2 = (up(s) for s in (f.up_latent0, f.up_latent1, f.up0, f.up1, f.up2))
        self.up_lr = (_phase_w(f.up_lowres.weight), f.up_lowres.bias.detach().half().repeat(4).contiguous())
        fw = f.fuse_lowres.weight.detach()
        c = fw.shape[1] // 2
        self.fuse_a, self.fuse_b, self.fuse_bias = _w1x1(fw[:, :c]), _w1x1(fw[:, c:]), f.fuse_lowres.bias
        self.dconv_w = [None] + [_cl(m.weight.detach()) for m in f.dconvs[1:]]
        self.res = []
        for fu in f.fusions:
            r = []
            for rb in (fu.resnet1, fu.resnet2):
                c1, c2 = rb.residual[1], rb.residual[3]
                r.append((_cl(c1.weight.detach()), c1.bias, _cl(c2.weight.detach()), c2.bias))
            self.res.append(r)
        self.fdec = [None] + [(_phase_w(m.weight), m.bias.detach().half().repeat(4).contiguous()) for m in f.fdeconv[1:]]
        self.out0 = (_w1x1(f.fusions[0].out_conv.weight), f.fusions[0].out_conv.bias)
        h = f.head
        cm = h.conv0.weight.shape[0] - 8          # 128 real channels; channel 128 = constant ones (deconv bias), 129..135 = 0
        self.h0 = (_cl(h.conv0.weight.detach()[:cm]), h.conv0.bias.detach()[:cm].contiguous())
        self.h0_const = h.conv0.bias.detach()[cm:].view(1, 8, 1, 1).clone()   # [1, 0, ..., 0]
        self.h1 = (_cl(h.conv1.weight.detach()), h.conv1.bias)
        wz = torch.zeros(8, h.conv2.weight.shape[1], 1, 1, dtype=torch.half, device=h.conv2.weight.device)
        wz[:4] = h.conv2.weight.detach()
        bz = torch.zeros(8, dtype=torch.half, device=wz.device)
        bz[:4] = h.conv2.bias.detach()
        self.h2 = (_cl(wz), bz)

    # ------------------------------------------------------------------------------------------------ helpers
    @staticmethod
    def deconv(x, wg, bias=None, res=None, out2=None, act2=None, act2_post=False):
        """2x2 / stride-2 ConvTranspose2d as one GEMM over the input pixels; the epilogue writes the 4 phases (NHWC)."""
        _, C, H, W = x.shape
        co = wg.shape[0] // 4
        y = torch.empty((1, co, 2 * H, 2 * W), device=x.device, dtype=torch.half, memory_format=torch.channels_last)
        omap = (W, 4 * W * co, 2 * co, co, 2 * W * co, co)
        return dconv(x, wg, bias, res=res, out=y, out2=out2, act2=act2, act2_post=act2_post, omap=omap)

    def up_chain(self, x, prm, last=None):
        y = dconv(x, prm[0])
        ws = prm[1]
        for i, w in enumerate(ws):
            y = self.deconv(y, w, **(last if (last and i == len(ws) - 1) else {}))
        return y

    @staticmethod
    def _nchw(rows, h):  # [h*h, C] contiguous -> [1, C, h, h] channels-last
        return rows.view(1, h, h, -1).permute(0, 3, 1, 2)

    def resblock(self, x, r, prm, dual=False):
        """x + conv2(relu(conv1(r))) with r = relu(x) given; dual: also return relu(result) (second epilogue output)."""
        w1, b1, w2, b2 = prm
        t = dconv(r, w1, b1, padding=1, act="relu")
        if not dual:
            return dconv(t, w2, b2, padding=1, res=x), None
        r2 = torch.empty_like(x)
        return dconv(t, w2, b2, padding=1, res=x, out2=r2, act2="relu", act2_post=True), r2

    def image_tokens(self, x2):
        G, _ = self.ivit.run(self.ivit.tokens_image(x2))
        return self.f.ie.norm(G[1:])

    def fov_tokens(self, x2):
        fv = self.f.fov
        Fz, _ = self.fvit.run(self.fvit.tokens_image(x2))
        return fv.encoder[1](fv.encoder[0].norm(Fz[1:]))                        # [576, 128]

    # ------------------------------------------------------------------------------------------------ network
    def forward(self, x, with_fov=True):
        f = self.f
        pe = f.pe
        x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        proj = pe.patch_embed.proj
        e = torch.cat([proj(x).flatten(2), proj(x1).flatten(2), proj(x2).flatten(2)], 2)[0].t()
        t = e[f.tok_idx]
        t = torch.cat([pe.cls_token.expand(t.shape[0], -1, -1), t], 1) + pe.pos_embed
        side = self.streams is not None
        if side:  # batch-1 ViTs concurrently on side streams
            main = torch.cuda.current_stream()
            s1, s2 = self.streams
            s1.wait_stream(main)
            with torch.cuda.stream(s1):
                g = self.image_tokens(x2)
            fz = None
            if with_fov and self.fvit is not None:
                s2.wait_stream(main)
                with torch.cuda.stream(s2):
                    fz = self.fov_tokens(x2)
        T, hk = self.pvit.run(t, hooks=set(f.hooks))
        f0 = pe.norm(T[f.m0]).view(-1, T.shape[1])
        f1 = pe.norm(T[f.m1]).view(-1, T.shape[1])
        f2 = pe.norm(T[f.m2]).view(-1, T.shape[1])
        l0 = hk[f.hooks[0]][f.m0].view(-1, T.shape[1])
        l1 = hk[f.hooks[1]][f.m0].view(-1, T.shape[1])
        if side:
            main.wait_stream(s1)
            g.record_stream(main)
        else:
            g = self.image_tokens(x2)
        n = f.n
        with fp16_accumulate(True):
            enc1 = self.up_chain(self._nchw(l1, 4 * n), self.up_l1)
            enc2 = self.up_chain(self._nchw(f0, 4 * n), self.up0)
            enc3 = self.up_chain(self._nchw(f1, 2 * n), self.up1)
            a = self.up_chain(self._nchw(f2, n), self.up2)
            b = self.deconv(self._nchw(g, n), *self.up_lr)
            enc4 = dconv(b, self.fuse_b, self.fuse_bias, res=dconv(a, self.fuse_a))
            # decoder
            r = torch.empty((1, 256, 2 * n, 2 * n), device=x.device, dtype=torch.half, memory_format=torch.channels_last)
            feat = dconv(enc4, self.dconv_w[4], padding=1, out2=r, act2="relu")
            lowres = feat
            x_, _ = self.resblock(feat, r, self.res[4][1])
            feat = self.deconv(x_, *self.fdec[4])
            encs = {3: enc3, 2: enc2, 1: enc1}
            del enc3, enc2, enc1, enc4, a, b
            for i in (3, 2, 1, 0):  # (references dropped as early as possible: the 768^2 maps are 302 MB each)
                r = torch.empty_like(feat)
                if i:
                    s = dconv(encs.pop(i), self.dconv_w[i], padding=1, res=feat, out2=r, act2="relu")
                else:  # latent0: its last deconvolution writes s = feat + latent0 and r = relu(latent0) (convs[0] = identity)
                    s = self.up_chain(self._nchw(l0, 4 * n), self.up_l0, last=dict(res=feat, out2=r, act2="relu"))
                del feat
                x_, r2 = self.resblock(s, r, self.res[i][0], dual=True)
                del s, r
                x_, _ = self.resblock(x_, r2, self.res[i][1])
                del r2
                feat = self.deconv(x_, *self.fdec[i]) if i else dconv(x_, *self.out0)
                del x_
            hb = torch.empty((1, self.h0[0].shape[0] + 8, *feat.shape[-2:]), device=x.device, dtype=torch.half,
                             memory_format=torch.channels_last)
            hb[:, -8:] = self.h0_const              # ones channel: conv0 then writes channels 0..127 only (128-wide tiles)
            dconv(feat, *self.h0, padding=1, out=hb[:, :-8])
            del feat
            h = hb
            h = dconv(h, *self.h1, padding=1, act="relu")
            z = dconv(h, *self.h2, act="relu")
            cid = F.pixel_shuffle(z[:, :4], 2)
        fov = None
        if with_fov and self.fvit is not None:
            fv = f.fov
            if side:
                main.wait_stream(s2)
                fz.record_stream(main)
            else:
                fz = self.fov_tokens(x2)
            y = fz.t().unsqueeze(0)                                              # [1, 128, 576]
            lr = fv.downsample(lowres.contiguous())
            fov = fv.head(y.reshape_as(lr) + lr)
        return cid, fov

    __call__ = forward
