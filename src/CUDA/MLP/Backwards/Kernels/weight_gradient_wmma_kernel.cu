#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
#define MATRIX_A 0
#define MATRIX_B 1
// ------------Setup----------------
// block tile
constexpr int BLOCK_TILE_M  = 128;  // block tile rows
constexpr int BLOCK_TILE_N  = 128;  // block tile cols
constexpr int BLOCK_TILE_K  = 16;   // block tile depth (K dimension)

// warp tile
constexpr int WARP_TILE_M   = 64;   // warp tile rows
constexpr int WARP_TILE_N   = 32;   // warp tile cols


// wmma tile
constexpr int WMMA_TILE_M = 16;
constexpr int WMMA_TILE_N = 16;
constexpr int WMMA_TILE_K = 16;

constexpr int NUM_WMMA_TILES_M = WARP_TILE_M / WMMA_TILE_M; // 4
constexpr int NUM_WMMA_TILES_N = WARP_TILE_N / WMMA_TILE_N; // 2
constexpr int NUM_WMMA_TILES_K = BLOCK_TILE_K / WMMA_TILE_K; // 1

// Number of warps in each block dimension
constexpr int WARPS_M       = BLOCK_TILE_M / WARP_TILE_M; // 2
constexpr int WARPS_N       = BLOCK_TILE_N / WARP_TILE_N; // 4
constexpr int NUM_WARPS     = WARPS_M * WARPS_N;          // 8
constexpr int BLOCK_THREADS = NUM_WARPS * 32;             // 256

// How many elements each thread reads into shared memory per K tile
constexpr int A_LOADS = (BLOCK_TILE_M * BLOCK_TILE_K) / BLOCK_THREADS; // 8
constexpr int B_LOADS = (BLOCK_TILE_K * BLOCK_TILE_N) / BLOCK_THREADS; // 8
// ----------------------------------

__device__ __forceinline__ void Load_Tile_opt(
         const float* __restrict__ Gradient,
         const float* __restrict__ PreviousActivation,
        __half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
        int M, int N, int K,
        int i0, int j0, int tid,
        int tile, int buffer);

template<typename AFrag, typename BFrag, typename CFrag>
__device__ __forceinline__ void Compute_Tile(__half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
                                             AFrag &A_fragments,
                                             BFrag &B_fragments,
                                             CFrag &Accumulated_fragments,
                                             int warp_row, int warp_col,
                                             int buffer);

template<typename AccFrag>
__device__ __forceinline__ void Write_Gradient_Tile(float* __restrict__ dW,
                                                    float norm_factor,
                                                    AccFrag& Accumulated_fragments,
                                                    float* C_Tile,
                                                    int M, int N,
                                                    int i0, int j0,
                                                    int warp_id,
                                                    int warp_row,
                                                    int warp_col);

template<typename AccFrag>
__device__ __forceinline__ void Write_Gradient_Tile_opt(float* __restrict__ dW,
                                                    float norm_factor,
                                                    AccFrag& Accumulated_fragments,
                                                    float* C_Tile,
                                                    int M, int N,
                                                    int i0, int j0,
                                                    int lane,
                                                    int warp_id,
                                                    int warp_row,
                                                    int warp_col);

extern "C" __global__ void weight_gradient_wmma_kernel(const float* __restrict__ Gradient,
                                                  const float* __restrict__ PreviousActivation,
                                                  float*       __restrict__ dW,
                                                  const float norm_factor,
                                                  int M, int N, int K)
{
// 1D array of threads
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;

    // thread within a warp
    const int lane     = tid & 31;

    // row and column indices for a warp within the block tile
    // warp_id = warp_col + (WARPS_N * warp_row)   (row-major)
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    // Global starting index of this block's output tile -> C[i0 +i,j0 + j]
    const int i0 = blockIdx.y * BLOCK_TILE_M;
    const int j0 = blockIdx.x * BLOCK_TILE_N;


    // store fp32 as fp16 to work with tensor cores
    // Adata is in column-major order
    // Bdata is in row-major order
    // pad the outer-most dimension by 8 to remove bank conflicts and keep 16 byte alignment
    __shared__ __align__(16) __half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8];

    // define wmma fragments 16x16x16 (M,N,K)
	wmma::fragment<wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K,
	               half,
	               wmma::col_major>
	               A_fragments[NUM_WMMA_TILES_M];

	wmma::fragment<wmma::matrix_b,
	               WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K,
	               half,
	               wmma::row_major>
	               B_fragments[NUM_WMMA_TILES_N];

	wmma::fragment<wmma::accumulator,
	               WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K,
	               float>
	               Accumulated_fragments[NUM_WMMA_TILES_M][NUM_WMMA_TILES_N];



    //initialise the accumulator fragment
    for (int tm = 0; tm < NUM_WMMA_TILES_M; tm ++)
        for (int tn = 0; tn < NUM_WMMA_TILES_N; tn ++)
        {
            wmma::fill_fragment(Accumulated_fragments[tm][tn], 0.0f);
        }


    const int NUMBER_OF_K_TILES = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

    int buffer = 0;
    //pre-load into buffer 0
    Load_Tile_opt(Gradient, PreviousActivation, ABdata, M, N, K, i0, j0, tid, 0, buffer);

    __syncthreads();

    for (int tile = 0; tile < NUMBER_OF_K_TILES; tile++) {
        // switch the buffers
        buffer = tile & 1;      // current buffer for compute
        const int next_buffer = buffer ^ 1;    // next buffer to load, toggles 1,0,...

        // load the next tile into the other buffer
        if (tile + 1 < NUMBER_OF_K_TILES)
            Load_Tile_opt(Gradient, PreviousActivation, ABdata, M, N, K, i0, j0, tid, tile + 1, next_buffer);

        // compute on the previously loaded tile
        Compute_Tile(ABdata, A_fragments, B_fragments, Accumulated_fragments,
                     warp_row, warp_col,
                     buffer);

        // ensure the next tile has loaded into the buffer
        __syncthreads();
    }

    float* C_Tile = reinterpret_cast<float*>(ABdata);

    Write_Gradient_Tile_opt(dW,norm_factor,
                        Accumulated_fragments,
                        C_Tile,
                        M, N,
                        i0, j0,
                        lane,
                        warp_id,
                        warp_row,
                        warp_col);

}

__device__ __forceinline__ void Load_Tile_opt(
         const float* __restrict__ Gradient,
         const float* __restrict__ PreviousActivation,
        __half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
        int M, int N, int K,
        int i0, int j0, int tid,
        int tile, int buffer)
{

    const int k_tile = tile * BLOCK_TILE_K;

    // Load Gradient matrix
    // BLOCK_TILE_K == 16 is a multiple of 8, such that float4 loads can be used,
    // i.e 8 loads per thread share the same global row of A.
    // -> 2 batches of float4 loads.
    // Alignment is guaranteed to be 16.
    #pragma unroll
    for (int i = 0; i < A_LOADS; i += 4)
    {
        const int lin0  = tid * A_LOADS + i;
        const int s_row = lin0 / BLOCK_TILE_K;
        const int s_col = lin0 % BLOCK_TILE_K;

        const int g_row = i0 + s_row;
        const int g_col = k_tile + s_col;

        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);

        if (g_row < M)
        {
            if (g_col + 3 < K)
            {
                v = *reinterpret_cast<const float4*>(&Gradient[g_row * K + g_col]);
            }
            else
            {
                // scalar fallback
                if (g_col + 0 < K) v.x = Gradient[g_row * K + g_col + 0];
                if (g_col + 1 < K) v.y = Gradient[g_row * K + g_col + 1];
                if (g_col + 2 < K) v.z = Gradient[g_row * K + g_col + 2];
                if (g_col + 3 < K) v.w = Gradient[g_row * K + g_col + 3];
            }
        }

        //Write into transposed shared memory
        // Convert float4 to 2x__half2 (round nearest) for wmma accumulation
        const __half2 h01 = __float22half2_rn(make_float2(v.x, v.y));
        const __half2 h23 = __float22half2_rn(make_float2(v.z, v.w));
        ABdata[MATRIX_A][buffer][s_col + 0][s_row] = __low2half(h01);
        ABdata[MATRIX_A][buffer][s_col + 1][s_row] = __high2half(h01);
        ABdata[MATRIX_A][buffer][s_col + 2][s_row] = __low2half(h23);
        ABdata[MATRIX_A][buffer][s_col + 3][s_row] = __high2half(h23);
    }

    // Load PreviousActivation matrix and transpose into shared memory
    #pragma unroll
    for (int i = 0; i < B_LOADS; i+=4)
    {
        const int lin0  = tid * B_LOADS + i;
        // For the transpose, the outer-most shared memory dimension is now BLOCK_TILE_K as B dimensions are N x K
        int s_row = lin0 / BLOCK_TILE_K;
        int s_col = lin0 % BLOCK_TILE_K;

        // Replace i0 with j0 because with respect to the blocks, rows have become columns
        const int g_row = j0 + s_row;

        // The tiling is now going across rows of B == traversing columns of transpose(B)
        const int g_col = k_tile + s_col;

        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);

        // Start the pointer at the beginning of this buffer's tile.
        __half* B_ptr = &ABdata[MATRIX_B][buffer][0][0];

        // row-major indexing
        if (g_row < N)
        {
                if (g_col +3 < K) // vectorised
                {
                    v = *reinterpret_cast<const float4*>(&PreviousActivation[g_row * K + g_col]);
                }
                else // scalar fallback
                {
                    if (g_col     < K) v.x = PreviousActivation[g_row * K + g_col];
                    if (g_col + 1 < K) v.y = PreviousActivation[g_row * K + g_col+1];
                    if (g_col + 2 < K) v.z = PreviousActivation[g_row * K + g_col+2];
                    if (g_col + 3 < K) v.w = PreviousActivation[g_row * K + g_col+3];
                }
        }

        // Write B shared data
        // We want to write column major indexing, respecting the padding, into the row-major buffer
        // to perform the transpose in shared memory.
        const int cm_index = s_col * (BLOCK_TILE_N + 8) + s_row;
        B_ptr[cm_index] = __float2half_rn(v.x);
        B_ptr[cm_index + (BLOCK_TILE_N + 8)] = __float2half_rn(v.y);
        B_ptr[cm_index + 2*(BLOCK_TILE_N + 8)] = __float2half_rn(v.z);
        B_ptr[cm_index + 3*(BLOCK_TILE_N + 8)] = __float2half_rn(v.w);
    }


//     #pragma unroll
//     for (int i = 0; i < B_LOADS; i+=4)
//     {
//         const int lin0  = tid * B_LOADS + i;
//         const int s_row = lin0 / BLOCK_TILE_N;
//         const int s_col = lin0 % BLOCK_TILE_N;
//
//         const int g_row = k_tile + s_row;
//         const int g_col = j0 + s_col;
//
//         float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
//
//         if (g_row < K)
//         {
//                 // scalar fallback , column major global indexing for transpose
//                 if (g_col + 0 < N) v.x = PreviousActivation[g_col * K + g_row];
//                 if (g_col + 1 < N) v.y = PreviousActivation[(g_col +1) * K + g_row];
//                 if (g_col + 2 < N) v.z = PreviousActivation[(g_col +2) * K + g_row];
//                 if (g_col + 3 < N) v.w = PreviousActivation[(g_col +3) * K + g_row];
//         }
//
//         // Write B shared data with vectorised half2 write
//         __half2* B_half2_ptr = reinterpret_cast<__half2*>(&ABdata[MATRIX_B][buffer][s_row][s_col]);
//         B_half2_ptr[0] = __float22half2_rn(make_float2(v.x, v.y));
//         B_half2_ptr[1] = __float22half2_rn(make_float2(v.z, v.w));
//     }
}

template<typename AFrag, typename BFrag, typename CFrag>
__device__ __forceinline__ void Compute_Tile(__half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
                                             AFrag &A_fragments,
                                             BFrag &B_fragments,
                                             CFrag &Accumulated_fragments,
                                             int warp_row, int warp_col,
                                             int buffer)
{
    const int w_row_tile = warp_row*WARP_TILE_M;
    const int w_col_tile = warp_col*WARP_TILE_N;

    #pragma unroll
    for (int k = 0; k < NUM_WMMA_TILES_K; k++)
    {
        const int k_ind = k*WMMA_TILE_K;

        #pragma unroll
        for (int m = 0; m < NUM_WMMA_TILES_M; m++)
        {
            const int row_ind = w_row_tile + m*WMMA_TILE_M;

            wmma::load_matrix_sync(A_fragments[m],
                                   &ABdata[MATRIX_A][buffer][k_ind][row_ind],
                                   BLOCK_TILE_M + 8);

        }
        #pragma unroll
        for (int n = 0; n < NUM_WMMA_TILES_N; n++)
        {
            const int col_ind = w_col_tile + n*WMMA_TILE_N;

            wmma::load_matrix_sync(B_fragments[n],
                                    &ABdata[MATRIX_B][buffer][k_ind][col_ind],
                                    BLOCK_TILE_N + 8);

        }
        #pragma unroll
         for (int m = 0; m < NUM_WMMA_TILES_M; m++)
         {
            #pragma unroll
             for (int n = 0; n < NUM_WMMA_TILES_N; n++)
                {
                    wmma::mma_sync(Accumulated_fragments[m][n],
                                   A_fragments[m], B_fragments[n],
                                   Accumulated_fragments[m][n]);
                }
         }
    }

}

template<typename AccFrag>
__device__ __forceinline__ void Write_Gradient_Tile_opt(float* __restrict__ dW,
                                                    float norm_factor,
                                                    AccFrag& Accumulated_fragments,
                                                    float* C_Tile,
                                                    int M, int N,
                                                    int i0, int j0,
                                                    int lane,
                                                    int warp_id,
                                                    int warp_row,
                                                    int warp_col)
{
    // Pad the leading dimension by 4 to suppress bank conflicts and ensure alignment
    constexpr int SMEM_STRIDE = WMMA_TILE_N + 4;
    float* dW_Tile_Warp_Ptr = C_Tile + warp_id * WMMA_TILE_M * SMEM_STRIDE;

    const int w_row_tile = i0 + warp_row * WARP_TILE_M;
    const int w_col_tile = j0 + warp_col * WARP_TILE_N;

   const int sub_tile_index = 8*lane;


    #pragma unroll
    for (int tile_row = 0; tile_row < NUM_WMMA_TILES_M; tile_row++)
    {
        const int row = w_row_tile + tile_row * WMMA_TILE_M;

        #pragma unroll
        for (int tile_col = 0; tile_col < NUM_WMMA_TILES_N; tile_col++)
        {
            const int col = w_col_tile + tile_col * WMMA_TILE_N;

            // if true, use float4 vectorised operations
            const bool full_tile = (row + WMMA_TILE_M <= M) && (col + WMMA_TILE_N <= N);
            const int TILE_NUM_ELEMENTS = Accumulated_fragments[tile_row][tile_col].num_elements;
            if (full_tile)
            {
            //apply normalisation (each lane in the warp computes up to 8 results)
            #pragma unroll
            for (int n=0; n < 8; n++)
            {
                const int ind = sub_tile_index + n;
                if (ind < TILE_NUM_ELEMENTS)
                {Accumulated_fragments[tile_row][tile_col].x[ind] *= norm_factor;}
            }

            float* dW_ptr = dW + row * N + col;

                wmma::store_matrix_sync(dW_ptr,
                                        Accumulated_fragments[tile_row][tile_col],
                                        N,
                                        wmma::mem_row_major);
            }
            else // write-back to shared memory -> global memory
            {

                // Store fragment into padded shared memory
                wmma::store_matrix_sync(dW_Tile_Warp_Ptr,
                                        Accumulated_fragments[tile_row][tile_col],
                                        SMEM_STRIDE,
                                        wmma::mem_row_major);

                const int ti = sub_tile_index / WMMA_TILE_N;
                const int tj = sub_tile_index % WMMA_TILE_N;
                const int offset = ti*N + tj;

                const int g_index = row*N + col;

                #pragma unroll
                for (int n =0; n < 2; n++)
                {
                    if ( (row + ti) < M)
                    {
                        #pragma unroll
                        for (int n = 0; n < 8; n+=4)
                        {
                        if ((col + tj + n + 3) < N) // vectorised
                        {
                            float4 W_vals = *reinterpret_cast<const float4*>(&dW_Tile_Warp_Ptr[ti*SMEM_STRIDE + tj + n]);
                            W_vals.x *= norm_factor;
                            W_vals.y *= norm_factor;
                            W_vals.z *= norm_factor;
                            W_vals.w *= norm_factor;

                            *reinterpret_cast<float4*>(&dW[g_index + offset + n]) = W_vals;
                        }
                        else // scalar
                        {
                            #pragma unroll
                            for (int p = 0; p < 4; p++)
                            {
                                if ((col + tj + n + p) < N)
                                {
                                dW[g_index + offset + n + p] = dW_Tile_Warp_Ptr[ti*SMEM_STRIDE + tj + n + p] * norm_factor;
                                }
                            }
                        }
                        }
                    }
                }
            }
        }
    }
}

template<typename AccFrag>
__device__ __forceinline__ void Write_Gradient_Tile(float* __restrict__ dW,
                                                    float norm_factor,
                                                    AccFrag& Accumulated_fragments,
                                                    float* C_Tile,
                                                    int M, int N,
                                                    int i0, int j0,
                                                    int warp_id,
                                                    int warp_row,
                                                    int warp_col)
{
    // Pad the leading dimension by 4 to suppress bank conflicts and ensure alignment
    constexpr int SMEM_STRIDE = WMMA_TILE_N + 4;
    float* dW_Tile_Warp_Ptr = C_Tile + warp_id * WMMA_TILE_M * SMEM_STRIDE;

    const int w_row_tile = i0 + warp_row * WARP_TILE_M;
    const int w_col_tile = j0 + warp_col * WARP_TILE_N;

    #pragma unroll
    for (int tile_row = 0; tile_row < NUM_WMMA_TILES_M; tile_row++)
    {
        const int row = w_row_tile + tile_row * WMMA_TILE_M;

        #pragma unroll
        for (int tile_col = 0; tile_col < NUM_WMMA_TILES_N; tile_col++)
        {
            const int col = w_col_tile + tile_col * WMMA_TILE_N;

            // Store fragment into padded shared memory
            wmma::store_matrix_sync(dW_Tile_Warp_Ptr,
                                    Accumulated_fragments[tile_row][tile_col],
                                    SMEM_STRIDE,
                                    wmma::mem_row_major);

            //
            // if true, use float4 vectorised operations
            const bool full_tile = (row + WMMA_TILE_M <= M) && (col + WMMA_TILE_N <= N);

            #pragma unroll
            for (int ti = 0; ti < WMMA_TILE_M; ti++)
            {
                const int g_row = row + ti;

                float* __restrict__ dW_row = dW + g_row * N;
                const float* dW_Tile_row = dW_Tile_Warp_Ptr + ti * SMEM_STRIDE;

                // If a full tile, use vectorised access
                if (full_tile)
                {
                    #pragma unroll
                    for (int tj = 0; tj < WMMA_TILE_N; tj += 4)
                    {
                        const float4 acc = *reinterpret_cast<const float4*>(dW_Tile_row + tj);

                        const float z0 = acc.x *norm_factor;
                        const float z1 = acc.y *norm_factor;
                        const float z2 = acc.z *norm_factor;
                        const float z3 = acc.w *norm_factor;

                        *reinterpret_cast<float4*>(dW_row + col + tj) = make_float4(z0,z1,z2,z3);
                    }
                }
                else
                {
                    // Scalar fallback
                    if (g_row < M)
                    {
                        #pragma unroll
                        for (int tj = 0; tj < WMMA_TILE_N; tj++)
                        {
                            const int g_col = col + tj;
                            if (g_col < N)
                            {
                                dW_row[g_col]  = dW_Tile_row[tj]*norm_factor;
                            }
                        }
                    }
                }
            }
        }
    }
}