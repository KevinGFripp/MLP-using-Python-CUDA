constexpr int BLOCK_SIZE_COL = 256;
constexpr int BLOCK_SIZE_VEC_COL = 256/4;
constexpr int NUM_WARPS = BLOCK_SIZE_COL/32;


extern "C" __global__ void bias_gradient_kernel(const float* __restrict__ layer_gradient,
                                                float* __restrict__ bias_gradient,
                                                const float norm_factor,
                                                int M,
                                                int N)
{
    // Number of blocks is always equal to number of rows, no bounds check required
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const int row_start = row*N;
    const int row_remainder = row_start % 4;

    const int col_vec_end = N / 4;
    const int remainder = N % 4;

    // shared memory to communicate between warps
    __shared__ float warp_wide_sum[32];

    const float* layer_gradient_ptr = layer_gradient + row_start;

    float sum = 0.0f;

    if (row_remainder == 0) // fully vectorised path + tail handling
    {
     //Each thread accumulates 4 floats in strides of the block size

        for (int vec_col = tid; vec_col < col_vec_end; vec_col += BLOCK_SIZE_VEC_COL)
        {
            const float4 sum_4 = *reinterpret_cast<const float4*>(&layer_gradient_ptr[4*vec_col]);
            sum += sum_4.x;
            sum += sum_4.y;
            sum += sum_4.z;
            sum += sum_4.w;
        }
        // single thread deals with tail
        if (tid == 0)
        {
            for (int n = 0 ; n< remainder; n++)
            {
                sum += layer_gradient_ptr[4*col_vec_end + n];
            }
        }
        // Accumulate with per-warp reduction
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

        int warp_id = tid / 32;
        int lane = tid % 32;

        // One thread writes warp-reduction to shared memory per warp_id
        if (lane == 0) warp_wide_sum[warp_id] = sum;

        __syncthreads();

        // Reduction on shared memory using first warp
        if (warp_id == 0)
        {
            sum = (lane < NUM_WARPS)? warp_wide_sum[lane] : 0.0f;

            // Accumulate with warp-reduction using the first warp
            sum += __shfl_down_sync(0xffffffff, sum, 16);
            sum += __shfl_down_sync(0xffffffff, sum, 8);
            sum += __shfl_down_sync(0xffffffff, sum, 4);
            sum += __shfl_down_sync(0xffffffff, sum, 2);
            sum += __shfl_down_sync(0xffffffff, sum, 1);

            if (lane == 0) bias_gradient[row] = norm_factor * sum;

        }

    }
    else
    {
        //scalar start reads
        if (tid == 0)
        {
            for (int n=0; n < row_remainder; n++)
            {
              if (n < N)
              {
                    sum += layer_gradient_ptr[n];
              }
            }
        }

        // base ptr + starting columns is now divisible by 4
        //Each thread accumulates 4 floats in strides of the block size
        for (int vec_col = tid + row_remainder; vec_col < col_vec_end; vec_col += BLOCK_SIZE_VEC_COL)
        {
            const float4 sum_4 = *reinterpret_cast<const float4*>(&layer_gradient_ptr[4*vec_col]);
            sum += sum_4.x;
            sum += sum_4.y;
            sum += sum_4.z;
            sum += sum_4.w;
        }

        // single thread deals with tail
        if (tid == 0)
        {
            for (int n = 0 ; n< remainder; n++)
            {
                sum += layer_gradient_ptr[4*col_vec_end + n];
            }
        }

        // Accumulate with per-warp reduction
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

        int warp_id = tid / 32;
        int lane = tid % 32;

        // One thread writes warp-reduction to shared memory per warp_id
        if (lane == 0) warp_wide_sum[warp_id] = sum;

        __syncthreads();

        // Reduction on shared memory using first warp
        if (warp_id == 0)
        {
            sum = (lane < NUM_WARPS)? warp_wide_sum[lane] : 0.0f;

            // Accumulate with warp-reduction using the first warp
            sum += __shfl_down_sync(0xffffffff, sum, 16);
            sum += __shfl_down_sync(0xffffffff, sum, 8);
            sum += __shfl_down_sync(0xffffffff, sum, 4);
            sum += __shfl_down_sync(0xffffffff, sum, 2);
            sum += __shfl_down_sync(0xffffffff, sum, 1);

            if (lane == 0) bias_gradient[row] = norm_factor * sum;
        }
    }
}