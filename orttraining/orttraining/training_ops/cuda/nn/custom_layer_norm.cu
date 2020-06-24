#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/nn/custom_layer_norm.cuh"

namespace cg = cooperative_groups;

/*
Fused bias add, residual (elementwise) add, and normalization layer.
Unlike the GELU, which doesn't require template parameters, this layer does since it
does rely fairly heavily on unrolling loops. Currently, I exclude bounds checks and
assume that the number of elements is a multiple of a power of 2. Default behavior
for our purposes uses 256 threads for floats, and 128 threads for __half. This restriction
is a result of using the shift parameter to perform the minimum number of register file
shuffles necessary, which requires the number of threads in the secondary reduction to
be 1, 2, 4, 8, 16, or 32. The number of threads here corresponds to the number of complete
warps in the threadblock.
For FP16, this kernel does not promote to FP32 in order to utilize the 2x throughput for
__half2 instructions, and avoid the conversion overhead (1/8 of __hal2 arithmetic).
For specific launch constraints, see the launch functions.
*/

template <int row_stride, int iterations>
__global__ void fused_bias_residual_layer_norm(float* vals,
                                               const float* residual,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               float* invvars = nullptr,
                                               float* means = nullptr)
{
    constexpr int iteration_stride = row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id / 32;

    float vals_arr[iterations];
    __shared__ float shr[iteration_stride >> 5];

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = residual[row * row_stride + i * iteration_stride + id];
        sum += vals_arr[i];
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];

    b.sync();

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += g.shfl_down(sum, i); }

    sum = g.shfl(sum, 0);
    float mean = sum / row_stride;
    if (g.thread_rank() == 0) means[row] = mean;
    float inv_variance, variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        variance += (vals_arr[i] - mean) * (vals_arr[i] - mean);
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) variance = shr[g.thread_rank()];

    b.sync();

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance += g.shfl_down(variance, i); }
    variance = g.shfl(variance, 0);
    variance /= row_stride;
    variance += epsilon;
    inv_variance = rsqrtf(variance);
    if (g.thread_rank() == 0) invvars[row] = inv_variance;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = (vals_arr[i] - mean) * inv_variance;
        vals_arr[i] =
            vals_arr[i] * gamma[i * iteration_stride + id] + beta[i * iteration_stride + id];
        vals[row * row_stride + i * iteration_stride + id] = vals_arr[i];
    }
}

template <int row_stride, int iterations>
__global__ void fused_bias_residual_layer_norm(__half* vals,
                                               const __half* residual,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               float* invvars = nullptr,
                                               float* means = nullptr)
{
#if __CUDA_ARCH__ >= 700
    constexpr int iteration_stride = row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;

    __half2 vals_arr[iterations];
    float2 vals_f[iterations];
    __shared__ float shr[iteration_stride >> 5];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_f[i] = __half22float2(residual_cast[row * row_stride + i * iteration_stride + id]);
        sum += vals_f[i].x;
        sum += vals_f[i].y;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];

    b.sync();

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += g.shfl_down(sum, i); }
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride * 2);

    float inv_variance, variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        variance += (vals_f[i].x - mean) * (vals_f[i].x - mean);
        variance += (vals_f[i].y - mean) * (vals_f[i].y - mean);
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) variance = shr[g.thread_rank()];

    b.sync();

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance += g.shfl_down(variance, i); }
    variance = g.shfl(variance, 0);
    variance /= (row_stride * 2);
    variance += epsilon;
    inv_variance = rsqrt(variance);

    __half2 mean_h = __float2half2_rn(mean);
    __half2 variance_h = __float2half2_rn(variance);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);

    if (g.thread_rank() == 0) {
        invvars[row] = inv_variance;
        means[row] = mean;
    }

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = __float22half2_rn(vals_f[i]);
        vals_arr[i] = (vals_arr[i] - mean_h) * h2rsqrt(variance_h);
        vals_arr[i] = vals_arr[i] * gamma_cast[i * iteration_stride + id] +
                      beta_cast[i * iteration_stride + id];
        vals_cast[row * row_stride + i * iteration_stride + id] = vals_arr[i];
    }
#endif
}

template <typename T, typename U>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     U epsilon,
                                     int64_t n1,
                                     int64_t n2,
                                     U* invvars,
                                     U* means);

template <>
void launch_bias_residual_layer_norm<float, float>(float* vals,
                                            const float* residual,
                                            const float* gamma,
                                            const float* beta,
                                            float epsilon,
                                            int64_t n1,
                                            int64_t n2,
                                            float* invvars,
                                            float* means)
{
    constexpr int threads = 256;

    dim3 grid_dim(n1);

    dim3 block_dim(threads);

    // There are some limitations to call below functions, now just enumerate the situations.
    if (n2 == 768)
        fused_bias_residual_layer_norm<768, 3><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 512)
        fused_bias_residual_layer_norm<512, 2><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 1024)
        fused_bias_residual_layer_norm<1024, 4><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 1536)
        fused_bias_residual_layer_norm<1536, 6><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 2048)
        fused_bias_residual_layer_norm<2048, 8><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 2560)
        fused_bias_residual_layer_norm<2560, 10><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <>
void launch_bias_residual_layer_norm<__half, float>(__half* vals,
                                             const __half* residual,
                                             const __half* gamma,
                                             const __half* beta,
                                             float epsilon,
                                             int64_t n1,
                                             int64_t n2,
                                             float* invvars,
                                             float* means)
{
    constexpr int threads = 128;

    dim3 grid_dim(n1);
    dim3 block_dim(threads);

    // There are some limitations to call below functions, now just enumerate the situations.
    if (n2 == 768)
        fused_bias_residual_layer_norm<384, 3><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 512)
        fused_bias_residual_layer_norm<256, 2><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 1024)
        fused_bias_residual_layer_norm<512, 4><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 1536)
        fused_bias_residual_layer_norm<768, 6><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 2048)
        fused_bias_residual_layer_norm<1024, 8><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else if (n2 == 2560)
        fused_bias_residual_layer_norm<1280, 10><<<grid_dim, block_dim, 0, 0>>>(
            vals, residual, gamma, beta, epsilon, invvars, means);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

