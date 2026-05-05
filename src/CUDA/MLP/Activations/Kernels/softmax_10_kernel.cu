constexpr int BLOCKSIZE = 256;
constexpr int COL_SIZE = 16;
constexpr int CLASSES = 10;
constexpr int PAD = 6;
constexpr int CLASSES_PLUS_PAD = CLASSES + PAD;
constexpr float PAD_VAL = -1e38f;

__device__ __forceinline__ float half_warp_max_reduce_and_broadcast(unsigned int mask, int base_lane, float value);
__device__ __forceinline__ float half_warp_sum_reduce_and_broadcast(unsigned int mask, int base_lane, float value);

extern "C" __global__ void softmax_10_kernel(const float* __restrict__ z,
                                             float* __restrict__ activation,
                                             int batch_size)
{
    const int tile_id = blockIdx.x * COL_SIZE;
    const int tid = threadIdx.x;

    const int row = tid / COL_SIZE;
    const int col = tid % COL_SIZE;

    const int warp_id = tid / 32;
    const int warp_lane = tid % 32;

    // every thread loads float4 values or writes padded values, and implicitly transposes
    __shared__ float tile[COL_SIZE][CLASSES_PLUS_PAD];

    // If a full tile fits within the number of columns, load without column access guards
    const bool full_tile = (tile_id + COL_SIZE) <= batch_size;

    if (row < 10 && col < 4)
    {
        const int vec_col = 4*col;
        const int col_id = tile_id + vec_col;

        if (full_tile)
        {
            float4 v = *reinterpret_cast<const float4*>(&z[row*batch_size + col_id]);
            tile[vec_col][row] = v.x;
            tile[vec_col+1][row] = v.y;
            tile[vec_col+2][row] = v.z;
            tile[vec_col+3][row] = v.w;
        }
        else
        {
            if (col_id + 3 < batch_size)
            {
                float4 v = *reinterpret_cast<const float4*>(&z[row*batch_size + col_id]);
                tile[vec_col][row] = v.x;
                tile[vec_col+1][row] = v.y;
                tile[vec_col+2][row] = v.z;
                tile[vec_col+3][row] = v.w;
            }
            else
            {
                tile[vec_col][row] = (col_id < batch_size)? z[row*batch_size + col_id] : PAD_VAL;
                tile[vec_col+1][row] = (col_id+1 < batch_size)? z[row*batch_size + col_id +1] : PAD_VAL;
                tile[vec_col+2][row] = (col_id+2 < batch_size)? z[row*batch_size + col_id +2] : PAD_VAL;
                tile[vec_col+3][row] = (col_id+3 < batch_size)? z[row*batch_size + col_id +3] : PAD_VAL;
            }
        }
    }
    else // 1 thread writes padding
    {
        tile[col][row] = PAD_VAL;
    }

    __syncthreads();

    // based on the warp_id, each warp controls the computation of two consecutive 16-element padded columns
    // Half a warp (0-15) does column 1, the other half (16-31) does column 2
    const int s_col = 2*warp_id;
    const int s_col_offset = warp_lane / 16;
    bool upper_half_warp = warp_lane >= 16;
    int base_lane = upper_half_warp? 16 : 0;
    // mask off threads 0-15 : 16-31
    unsigned int mask = upper_half_warp? 0xFFFF0000 : 0x0000FFFF;

    float* tile_ptr = &tile[s_col + s_col_offset][0];

    const float out_val = tile_ptr[warp_lane - base_lane];
    const float max_val = half_warp_max_reduce_and_broadcast(mask, base_lane, out_val);

    const float exp_val = __expf(out_val - max_val);
    const float exp_sum = half_warp_sum_reduce_and_broadcast(mask, base_lane, exp_val);

    const int g_row = warp_lane - base_lane;
    const int g_col = tile_id + s_col + s_col_offset;

    // write back to global memory
    if (g_row < 10 && g_col < batch_size) activation[g_row*batch_size + g_col] = exp_val/exp_sum;

}

__device__ __forceinline__ float half_warp_max_reduce_and_broadcast(unsigned int mask, int base_lane, float value)
{
// half-warp max reduction
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
        value = fmaxf(value, __shfl_down_sync(mask, value, offset, CLASSES_PLUS_PAD));

    //Broadcast max reduction from base lane
    float max_val = __shfl_sync(mask, value, base_lane, CLASSES_PLUS_PAD);

    return max_val;
}

__device__ __forceinline__ float half_warp_sum_reduce_and_broadcast(unsigned int mask, int base_lane, float value)
{
// half-warp sum reduction
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
        value += __shfl_down_sync(mask, value, offset, CLASSES_PLUS_PAD);

    //Broadcast max reduction from base lane
    float sum = __shfl_sync(mask, value, base_lane, CLASSES_PLUS_PAD);

    return sum;
}