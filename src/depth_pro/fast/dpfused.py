"""Loader for csrc/dp_fused.cu (CUTLASS conv2d with the generalized fused epilogue used by the Depth Pro cuda-kernels path):
compiled once per source hash into prebuilt/dpf_<hash>/ (SASS sm_80/86/89/120 + PTX), loaded with
torch.ops.load_library afterwards (no compiler at run time; nothing compiled or tuned per input size). Op: torch.ops.dpf.conv.
The shipped .so is built for torch 2.14 + CUDA 13.0 (SASS sm_80/86/89/120 + PTX). For another torch build delete prebuilt/dpf_*
and rebuild once (nvcc matching torch, CUTLASS headers): CUTLASS_DIR=/path/to/cutlass python -m depth_pro.fast.dpfused
"""
import hashlib
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "csrc", "dp_fused.cu")
CUTLASS = os.environ.get("CUTLASS_DIR", "")
_loaded = [False]


def load():
    if _loaded[0]:
        return
    tag = hashlib.sha1(open(SRC, "rb").read()).hexdigest()[:12]
    bdir = os.path.join(HERE, "prebuilt", f"dpf_{tag}")
    so = os.path.join(bdir, f"dpf_{tag}.so")
    if not os.path.exists(so):
        os.makedirs(bdir, exist_ok=True)
        from torch.utils.cpp_extension import load as _load
        libdir = os.path.join(HERE, "prebuilt", "lib")  # pip CUDA ships only libcudart.so.13: give the linker a libcudart.so
        os.makedirs(libdir, exist_ok=True)
        cudart = os.path.join(os.environ.get("CUDA_HOME", ""), "lib", "libcudart.so.13")
        if os.path.exists(cudart) and not os.path.exists(os.path.join(libdir, "libcudart.so")):
            os.symlink(cudart, os.path.join(libdir, "libcudart.so"))
        arch = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in (80, 86, 89, 120)] + ["-gencode=arch=compute_80,code=compute_80"]
        _load(name=f"dpf_{tag}", sources=[SRC], build_directory=bdir, verbose=False, is_python_module=False,
              extra_include_paths=[os.path.join(CUTLASS, "include")], extra_ldflags=[f"-L{libdir}"],
              extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-U__CUDA_NO_HALF_OPERATORS__",
                                 "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", *arch])
    torch.ops.load_library(so)
    _loaded[0] = True


ACT = {None: 0, "relu": 1, "gelu": 5}


def conv(x, w, bias=None, scale=None, res=None, padding=0, stride=1, act=None, out=None, out2=None, act2=None, act2_post=False,
         omap=(), cfg=-1):
    load()
    p = [padding, padding] if isinstance(padding, int) else list(padding)
    s = [stride, stride] if isinstance(stride, int) else list(stride)
    return torch.ops.dpf.conv(x, w, bias, scale, res, p, s, ACT[act], out, out2, ACT[act2], int(act2_post), list(omap), cfg)


if __name__ == "__main__":
    load()
    print("ok", torch.ops.dpf.conv)
