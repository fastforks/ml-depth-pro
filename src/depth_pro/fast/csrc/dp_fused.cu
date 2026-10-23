// exp 004 Depth Pro max path: CUTLASS implicit-GEMM conv2d (sm_80-style mma.sync, fp16 accumulation) with a generalized fused
// epilogue. Mainloop / tile configs / tile rule = exp 006 fastbb csrc/fused_conv.cu (vendored from exp 008); only the epilogue is new.
// Compiled once (problem size = runtime argument; nothing compiled or tuned per input size).
//
//   v = acc (+ bias[n]) (* scale[n])          scale = LayerScale gamma of a ViT branch
//   y2 = act2(v)                              optional second output, before the residual (act2_post = 0) ...
//   v += res                                  optional residual (same layout as y)
//   y = act(v);  y2 = act2(v) (act2_post = 1) ... or after it
// Output addressing (omap): GEMM row m = output pixel (q fastest), column n = output channel:
//   offset(m, n) = (m / Q) * rowst + (m % Q) * pixst + col(n),  col(n) = n  or, with phk > 0 (2x2 stride-2 deconvolution as one
//   GEMM over input pixels with N = 4 * phk columns): ph = n / phk -> (ph >> 1) * ph_row + (ph & 1) * ph_col + n % phk.
//   The residual and y2 use the same map. Default map: Q = 1, rowst = ldy (channel slice of a wider NHWC buffer).
// x, res, y, y2: NHWC (channels-last) fp16; w: KRSC fp16; C % 8 == 0, K % 8 == 0, dilation 1, groups 1.
// act / act2: 0 none, 1 relu, 5 gelu(erf), 6 gelu(tanh)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

struct EpiArgs {
    const __half* bias;
    const __half* scale;
    const __half* res;
    __half* y;
    __half* y2;
    int act, act2, act2_post;
    int Q, phk;
    long long rowst, pixst, ph_row, ph_col;
};

__device__ __forceinline__ float act_f(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 5: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
        case 6: return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
    return v;
}

__device__ __forceinline__ long long out_off(EpiArgs const& a, int m, int n) {
    long long o = (long long)(m / a.Q) * a.rowst + (long long)(m % a.Q) * a.pixst;
    if (a.phk) {
        int ph = n / a.phk;
        o += (ph >> 1) * a.ph_row + (ph & 1) * a.ph_col + (n - ph * a.phk);
    } else {
        o += n;
    }
    return o;
}

template <class OutIt, bool FULL>
struct Visitor {
    static int const kIterations = OutIt::kIterations;
    static int const kElementsPerAccess = OutIt::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "8 channels per access");
    using ElementAccumulator = Hf;
    using AccumulatorFragment = cutlass::Array<Hf, kElementsPerAccess>;
    using ThreadMap = typename OutIt::ThreadMap;
    struct SharedStorage {};
    static int const kFrags = ThreadMap::Iterations::kCount;

    EpiArgs const& a;
    OutIt it;
    int M, N;
    uint4 pr[kFrags];

    CUTLASS_DEVICE Visitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                           int thread_idx, cutlass::MatrixCoord tb_offset)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()) {}

    CUTLASS_DEVICE long long off_of(int p, int n) const {
        if constexpr (FULL) return out_off(a, p, n);
        else return (long long)p * a.rowst + n;
    }
    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
        if (!a.res) return;
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(i);
            int p = co.row();
            int n = co.column();
            if (p < M && n < N) pr[i] = __ldg(reinterpret_cast<const uint4*>(a.res + off_of(p, n)));
        }
    }
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void end_step(int) { ++it; }
    CUTLASS_DEVICE void end_epilogue() {}

    CUTLASS_DEVICE void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
        cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(frag_idx);
        int p = co.row();
        int n = co.column();
        if (p >= M || n >= N) return;
        long long off = off_of(p, n);
        const __half* av = reinterpret_cast<const __half*>(&acc);
        __align__(16) __half bv[8];
        __align__(16) __half sv[8];
        if (a.bias) *reinterpret_cast<uint4*>(bv) = __ldg(reinterpret_cast<const uint4*>(a.bias + n));
        if (a.scale) *reinterpret_cast<uint4*>(sv) = __ldg(reinterpret_cast<const uint4*>(a.scale + n));
        const __half* rv = reinterpret_cast<const __half*>(&pr[frag_idx]);
        __align__(16) __half o[8];
        __align__(16) __half o2[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(av[k]);
            if (a.bias) v += __half2float(bv[k]);
            if (a.scale) v *= __half2float(sv[k]);
            if constexpr (FULL) if (!a.act2_post) o2[k] = __float2half(act_f(v, a.act2));
            if (a.res) v += __half2float(rv[k]);
            o[k] = __float2half(act_f(v, a.act));
            if constexpr (FULL) if (a.act2_post) o2[k] = __float2half(act_f(v, a.act2));
        }
        *reinterpret_cast<uint4*>(a.y + off) = *reinterpret_cast<const uint4*>(o);
        if constexpr (FULL) if (a.y2) *reinterpret_cast<uint4*>(a.y2 + off) = *reinterpret_cast<const uint4*>(o2);
    }
};

template <class TB, class WS, int Stages, int Swz, bool FULL>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt, FULL>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        V, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
};

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) dp_conv_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    cutlass::MatrixCoord extent(prm.problem_size.N * prm.problem_size.P * prm.problem_size.Q, prm.problem_size.K);
    typename F::V v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run(cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e, cudaStream_t s) {
    using Base = typename F::Base;
    typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({p.N, p.H, p.W, p.C})}, {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
                                  {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(dp_conv_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return 2;
        attr_set = true;
    }
    dp_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

// cfg ids = fastbb fused_conv.cu ids (subset); FULL = output map / second output (decoder), else the light epilogue (ViT GEMMs)
template <bool FULL>
static int dispatch_t(int cfg, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                      cudaStream_t s) {
    switch (cfg) {
        case 2: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1, FULL>>(p, x, w, e, s);
        case 7: return run<Fused<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4, FULL>>(p, x, w, e, s);
        case 8: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4, FULL>>(p, x, w, e, s);
        case 9: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4, FULL>>(p, x, w, e, s);
        case 13: return run<Fused<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4, FULL>>(p, x, w, e, s);
    }
    return -1;
}

static int dispatch(int cfg, bool full, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w,
                    EpiArgs const& e, cudaStream_t s) {
    return full ? dispatch_t<true>(cfg, p, x, w, e, s) : dispatch_t<false>(cfg, p, x, w, e, s);
}

static const int SMS = [] { int d = 0, n = 36; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); return n; }();

// = fastbb pick() (fixed rule, no timing) + narrow-K tile for K <= 32
static int pick(long long M, int K) {
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    auto fits = [&](int tn) { return ((K + tn - 1) / tn) * tn <= 1.1 * K; };
    if (K <= 32) return 13;
    if (fits(128)) {
        if (tiles(128, 128) >= 4 * SMS && M >= 4096) return 8;
        if (tiles(64, 128) >= SMS) return 7;    // small M (batch-1 ViT GEMMs, M = 577): row-tile waste + wave quantization
        return 2;
    }
    if (tiles(128, 64) >= SMS) return 9;
    return 2;
}

static const __half* hp(c10::optional<at::Tensor> const& t) {
    return t ? reinterpret_cast<const __half*>(t->data_ptr()) : nullptr;
}

static bool same_layout(at::Tensor const& a, at::Tensor const& y) {
    if (a.sizes() != y.sizes()) return false;
    if (a.strides() == y.strides()) return true;
    return a.is_contiguous(at::MemoryFormat::ChannelsLast) && y.is_contiguous(at::MemoryFormat::ChannelsLast);
}

at::Tensor dp_conv(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> scale,
                   c10::optional<at::Tensor> res, std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act,
                   c10::optional<at::Tensor> out, c10::optional<at::Tensor> out2, int64_t act2, int64_t act2_post,
                   std::vector<int64_t> omap, int64_t cfg) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf, "fp16 CUDA tensors");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "x channels-last");
    TORCH_CHECK(w.is_contiguous(at::MemoryFormat::ChannelsLast), "weight channels-last");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int K = w.size(0), R = w.size(2), S = w.size(3);
    TORCH_CHECK(C % 8 == 0 && K % 8 == 0 && w.size(1) == C, "C, K % 8");
    int ph = padding[0], pw = padding[1], sh = stride[0], sw = stride[1];
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    at::Tensor y;
    EpiArgs e{hp(bias), hp(scale), nullptr, nullptr, nullptr, (int)act, (int)act2, (int)act2_post, 1, 0, K, 0, 0, 0};
    if (!omap.empty()) {
        TORCH_CHECK(out && omap.size() == 6 && N == 1, "omap needs out, 6 values, batch 1");
        y = *out;
        e.Q = (int)omap[0]; e.rowst = omap[1]; e.pixst = omap[2]; e.phk = (int)omap[3]; e.ph_row = omap[4]; e.ph_col = omap[5];
        TORCH_CHECK(e.Q == Q && (e.phk == 0 || e.phk % 8 == 0) && e.rowst % 8 == 0 && e.pixst % 8 == 0, "omap");
    } else if (out) {
        y = *out;
        int ldy = y.stride(3);
        TORCH_CHECK(y.size(0) == N && y.size(1) == K && y.size(2) == P && y.size(3) == Q && y.stride(1) == 1 &&
                    y.stride(2) == (long long)Q * ldy && y.stride(0) == (long long)P * Q * ldy && ldy % 8 == 0, "out layout");
        e.rowst = ldy;
    } else {
        y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    }
    TORCH_CHECK(((uintptr_t)y.data_ptr()) % 16 == 0 && y.scalar_type() == at::kHalf, "out alignment");
    e.y = reinterpret_cast<__half*>(y.data_ptr());
    if (res) {
        TORCH_CHECK(res->scalar_type() == at::kHalf && ((uintptr_t)res->data_ptr()) % 16 == 0, "res");
        if (omap.empty()) TORCH_CHECK(same_layout(*res, y), "res layout = out layout");
        e.res = hp(res);
    }
    if (out2) {
        TORCH_CHECK(out2->scalar_type() == at::kHalf && ((uintptr_t)out2->data_ptr()) % 16 == 0, "out2");
        if (omap.empty()) TORCH_CHECK(same_layout(*out2, y), "out2 layout = out layout");
        e.y2 = reinterpret_cast<__half*>(out2->data_ptr());
    }
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {1, 1}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    if (cfg < 0) cfg = pick((long long)N * P * Q, K);
    bool full = !omap.empty() || out2.has_value();
    int rc = dispatch((int)cfg, full, p, x.data_ptr(), w.data_ptr(), e, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(rc == 0, "dp_conv failed rc=", rc, " cfg=", cfg, " x=", x.sizes(), " w=", w.sizes());
    return y;
}

TORCH_LIBRARY(dpf, m) {
    m.def("conv(Tensor x, Tensor w, Tensor? bias, Tensor? scale, Tensor? res, int[] padding, int[] stride, int act, "
          "Tensor? out, Tensor? out2, int act2, int act2_post, int[] omap, int cfg) -> Tensor");
}
TORCH_LIBRARY_IMPL(dpf, CUDA, m) { m.impl("conv", dp_conv); }
