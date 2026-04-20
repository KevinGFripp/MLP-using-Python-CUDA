__device__ __forceinline__ float4 ReLU_float4(float4 z);
__device__ __forceinline__ float ReLU(float z);

extern "C" __global__ void ReLU_kernel(float* __restrict__ z, float* __restrict__ a, const int size)
{

    const int matrix_end = size/4;
    const int remainder = size - 4*matrix_end;

    const int tid = blockDim.x*blockIdx.x + threadIdx.x;

    //vectorised
    if(tid < matrix_end)
    {
        float4* z_ptr = reinterpret_cast<float4*>(z);
        float4* a_ptr = reinterpret_cast<float4*>(a);

        a_ptr[tid] = ReLU_float4(z_ptr[tid]);

    }
    //remainder tail
    if (tid == 0)
    {
        for (int n = 0; n < remainder; n++)
        {
            const float z_val = z[4*matrix_end +n];
            a[4*matrix_end +n] = ReLU(z_val);

        }
    }

}

__device__ __forceinline__ float4 ReLU_float4(float4 z)
{
    return make_float4((z.x > 0.0f)? z.x : 0.0f,
                       (z.y > 0.0f)? z.y : 0.0f,
                       (z.z > 0.0f)? z.z : 0.0f,
                       (z.w > 0.0f)? z.w : 0.0f);
}

__device__ __forceinline__ float ReLU(float z)
{
    return (z > 0.0f)? z : 0.0f;
}