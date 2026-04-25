//---------------------------------------------------------------------
// Operator definitions for float4
__device__ __forceinline__ float4 operator+(float4 a, float4 b)
{ return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);}
__device__ __forceinline__ float4 operator+(float4 a, float b)
{ return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);}
__device__ __forceinline__ float4 operator+(float a, float4 b)
{ return make_float4(a+ b.x, a+ b.y, a+ b.z, a+ b.w);}


__device__ __forceinline__ float4 operator-(float4 a, float4 b)
{ return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);}
__device__ __forceinline__ float4 operator-(float4 a, float b)
{ return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);}
__device__ __forceinline__ float4 operator-(float a, float4 b)
{ return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);}


__device__ __forceinline__ float4 operator*(float4 a, float4 b)
{ return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);}
__device__ __forceinline__ float4 operator*(float4 a, float b)
{ return make_float4(a.x*b, a.y*b, a.z*b, a.w*b);}
__device__ __forceinline__ float4 operator*(float a, float4 b)
{ return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);}


__device__ __forceinline__ float4 operator/(float4 a, float4 b)
{ return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);}
__device__ __forceinline__ float4 operator/(float4 a, float b)
{const float inv_b = 1./b;
 return make_float4(inv_b*a.x, inv_b*a.y, inv_b*a.z, inv_b*a.w);}
 __device__ __forceinline__ float4 operator/(float a, float4 b)
{ return make_float4(a/b.x, a/b.y, a/b.z, a/b.w);}
//------------------------------------------------------------------------

// reciprocal square root
__device__ __forceinline__ float4 r_sqrtf_float4(float4 a)
{ return make_float4(1.f/(sqrtf(a.x)+1e-8),
                     1.f/(sqrtf(a.y)+1e-8),
                     1.f/(sqrtf(a.z)+1e-8),
                     1.f/(sqrtf(a.w)+1e-8));}



extern "C" __global__ void adam_optimiser_vectorised_kernel(float* __restrict__ Weights,
                                                            float* __restrict__ Biases,
                                                            float* __restrict__ WeightGradients,
                                                            float* __restrict__ BiasGradients,
                                                            float* __restrict__ mWeights,
                                                            float* __restrict__ mbiases,
                                                            float* __restrict__ vWeights,
                                                            float* __restrict__ vbiases,
                                                            const float learning_rate,
                                                            const float beta1,
                                                            const float beta2,
                                                            const float beta1_correction,
                                                            const float beta2_correction,
                                                            const int t,
                                                            int M, int N)
{

 const int tid = blockDim.x*blockIdx.x + threadIdx.x;

 const int matrix_end = M * N / 4;
 const int matrix_end_remainder = M*N - 4*matrix_end;

 const int vec_end = M / 4;
 const int vec_end_remainder = M - 4*vec_end;

 const int base_mat_ind = 4*matrix_end;
 const int base_vec_ind = 4*vec_end;

 const float prefactor1 = 1.f - beta1;
 const float prefactor2 = 1.f - beta2;

 const float eps = 1e-8f;

 // vectorised weights update
 if (tid < matrix_end)
 {
    const int g_id = 4*tid;

    // vectorised loads
    const float4 Wgrad = *reinterpret_cast<float4*>(&WeightGradients[g_id]);
    const float4 mW = *reinterpret_cast<float4*>(&mWeights[g_id]);
    const float4 vW = *reinterpret_cast<float4*>(&vWeights[g_id]);

    // momentum
    const float4 update_mW = beta1 * mW + prefactor1 * Wgrad;
    *reinterpret_cast<float4*>(&mWeights[g_id]) = update_mW;

    // 2nd moment
    const float4 update_vW = beta2 * vW + prefactor2 * Wgrad * Wgrad;
    *reinterpret_cast<float4*>(&vWeights[g_id]) = update_vW;

    // bias correction
    const float4 mW_hat = beta1_correction * update_mW;
    const float4 vW_hat = beta2_correction * update_vW;

    // final update
    const float4 Weights_float4 = *reinterpret_cast<float4*>(&Weights[g_id]);
    *reinterpret_cast<float4*>(&Weights[g_id]) = Weights_float4 -(learning_rate * mW_hat * r_sqrtf_float4(vW_hat));
 }

 // handle remainder tail
 if (tid == 0)
 {
    for (int n=0; n < matrix_end_remainder; n++)
    {
    const float Wgrad_f = WeightGradients[base_mat_ind + n];

    // momentum
    const float update_mW_f = beta1 * mWeights[base_mat_ind + n] + prefactor1 * Wgrad_f;
        mWeights[base_mat_ind + n] = update_mW_f;

    // 2nd moment
    const float update_vW_f = beta2 * vWeights[base_mat_ind + n] + prefactor2 * Wgrad_f * Wgrad_f;
        vWeights[base_mat_ind + n] = update_vW_f;

    // bias correction
    const float mW_hat_f = beta1_correction * update_mW_f;
    const float vW_hat_f = beta2_correction * update_vW_f;

    // final update
    Weights[base_mat_ind + n] -= learning_rate * mW_hat_f/(sqrtf(vW_hat_f) + eps);
    }
 }

// vectorised biases update
if (tid < vec_end)
{
    const int g_id = 4*tid;

    // vectorised loads
    const float4 bgrad = *reinterpret_cast<float4*>(&BiasGradients[g_id]);
    const float4 mb = *reinterpret_cast<float4*>(&mbiases[g_id]);
    const float4 vb = *reinterpret_cast<float4*>(&vbiases[g_id]);

    // momentum
    const float4 update_mb = beta1 * mb + prefactor1 * bgrad;
    *reinterpret_cast<float4*>(&mbiases[g_id]) = update_mb;

    // 2nd moment
    const float4 update_vb = beta2 * vb + prefactor2 * bgrad * bgrad;
    *reinterpret_cast<float4*>(&vbiases[g_id]) = update_vb;

    // bias correction
    const float4 mb_hat = beta1_correction * update_mb;
    const float4 vb_hat = beta2_correction * update_vb;

    // final update
    const float4 Biases_float4 = *reinterpret_cast<float4*>(&Biases[g_id]);
    *reinterpret_cast<float4*>(&Biases[g_id]) = Biases_float4 -(learning_rate * mb_hat * r_sqrtf_float4(vb_hat));

}

// handle remainder tail
 if (tid == 0)
 {
    for (int n=0; n < vec_end_remainder; n++)
    {
    const float bgrad_f = BiasGradients[base_vec_ind + n];

    // momentum
    const float update_mb_f = beta1 * mbiases[base_vec_ind + n] + prefactor1 * bgrad_f;
        mbiases[base_vec_ind + n] = update_mb_f;

    // 2nd moment
    const float update_vb_f = beta2 * vbiases[base_vec_ind + n] + prefactor2 * bgrad_f * bgrad_f;
        vbiases[base_vec_ind + n] = update_vb_f;

    // bias correction
    const float mb_hat_f = beta1_correction * update_mb_f;
    const float vb_hat_f = beta2_correction * update_vb_f;

    // final update
    Biases[base_vec_ind + n] -= learning_rate * mb_hat_f/(sqrtf(vb_hat_f) + eps);
    }
 }

 }

















