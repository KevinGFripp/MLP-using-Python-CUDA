
extern "C" __global__ void adam_optimiser_kernel(float* __restrict__ Weights,
                                                 float* __restrict__ Biases,
                                                const float* __restrict__ WeightGradients,
                                                const float* __restrict__ BiasGradients,
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

 const int i = blockDim.y*blockIdx.y + threadIdx.y;
 const int j = blockDim.x*blockIdx.x + threadIdx.x;

 const float prefactor1 = 1. - beta1;
 const float prefactor2 = 1. - beta2;

 const float eps = 1e-8f;

 // update weights
 if (i < M && j < N)
 {
    const float Wgrad = WeightGradients[i*N + j];

    // momentum
    const float update_mW = beta1 * mWeights[i*N + j] + prefactor1 * Wgrad;
        mWeights[i*N + j] = update_mW;

    // 2nd moment
    const float update_vW = beta2 * vWeights[i*N + j] + prefactor2 * Wgrad * Wgrad;
        vWeights[i*N + j] = update_vW;

    // bias correction
    const float mW_hat = beta1_correction * update_mW;
    const float vW_hat = beta2_correction * update_vW;

    // final update
    Weights[i*N + j] -= learning_rate * mW_hat/(sqrtf(vW_hat) + eps);

 }

 //update biases
 if ( j == 0 && i < M)
 {
 const float bgrad = BiasGradients[i];

 // momentum
 const float update_mb = beta1 * mbiases[i] + prefactor1 * bgrad;
            mbiases[i] = update_mb;

 // 2nd moment
 const float update_vb = beta2 * vbiases[i] + prefactor2 * bgrad * bgrad;
            vbiases[i] = update_vb;

 // bias correction
 const float mb_hat = beta1_correction * update_mb;
 const float vb_hat = beta2_correction * update_vb;

 // final update
 Biases[i] -= learning_rate * mb_hat/(sqrtf(vb_hat) + eps);
 }

}
