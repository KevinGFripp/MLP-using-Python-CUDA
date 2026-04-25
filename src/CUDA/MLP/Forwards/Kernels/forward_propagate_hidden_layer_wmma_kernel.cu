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
        const float* __restrict__ A,
        const float* __restrict__ B,
        __half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
        int M, int N, int K,
        int i0, int j0, int tid,
        int tile, int buffer);

__device__ __forceinline__ void Load_Tile( const float* __restrict__ A,const float* __restrict__ B,
                                          __half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
                                           int M, int N, int K,
                                           int i0,int j0,int tid,
                                           int tile,int buffer);

template<typename AFrag, typename BFrag, typename CFrag>
__device__ __forceinline__ void Compute_Tile(__half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
                                             AFrag &A_fragments,
                                             BFrag &B_fragments,
                                             CFrag &Accumulated_fragments,
                                             int warp_row, int warp_col,
                                             int buffer);

template<typename AccFrag>
__device__ __forceinline__ void Write_Tile(float* C, AccFrag &Accumulated_fragments,
                                           int M, int N,
                                           int i0, int j0,
                                           int warp_row, int warp_col);

__device__ __forceinline__ float ReLU(float z);

template<typename AccFrag>
__device__ __forceinline__ void Write_Activation_Tile(float* __restrict__ C,
                                                      const float* __restrict__ Bias,
                                                      float* __restrict__ D,
                                                      AccFrag& Accumulated_fragments,
                                                      float* C_Tile,
                                                      int M, int N,
                                                      int i0, int j0,
                                                      int warp_id,
                                                      int warp_row,
                                                      int warp_col);

template<typename AccFrag>
__device__ __forceinline__ void Write_Activation_Tile_opt(float* __restrict__ C,
                                             const float* __restrict__ Bias,
                                             float* __restrict__ D,
                                             AccFrag& Accumulated_fragments,
                                             float* C_Tile,
                                             int M, int N,
                                             int i0, int j0,
                                             int lane,
                                             int warp_id,
                                             int warp_row,
                                             int warp_col);

extern "C" __global__ void forward_propagate_hidden_layer_wmma_kernel(const float* __restrict__ A,
                                                                      const float* __restrict__ B,
                                                                      float*       __restrict__ C,
                                                                      const float* __restrict__ b,
                                                                      float*       __restrict__ D,
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
    //__shared__ __align__(16) __half Bdata[2][BLOCK_TILE_K][BLOCK_TILE_N +8];

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
    Load_Tile_opt(A, B, ABdata, M, N, K, i0, j0, tid, 0, buffer);

    __syncthreads();

    for (int tile = 0; tile < NUMBER_OF_K_TILES; tile++) {
        // switch the buffers
        buffer = tile & 1;      // current buffer for compute
        const int next_buffer = buffer ^ 1;    // next buffer to load, toggles 1,0,...

        // load the next tile into the other buffer
        if (tile + 1 < NUMBER_OF_K_TILES)
            Load_Tile_opt(A, B, ABdata, M, N, K, i0, j0, tid, tile + 1, next_buffer);

        // compute on the previously loaded tile
        Compute_Tile(ABdata, A_fragments, B_fragments, Accumulated_fragments,
                     warp_row, warp_col,
                     buffer);

        // ensure the next tile has loaded into the buffer
        __syncthreads();
    }

    float* C_Tile = reinterpret_cast<float*>(ABdata);

    Write_Activation_Tile_opt(C, b, D,
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
        const float* __restrict__ A,
        const float* __restrict__ B,
        __half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
        int M, int N, int K,
        int i0, int j0, int tid,
        int tile, int buffer)
{


    const int k_tile = tile * BLOCK_TILE_K;

    // Load A matrix
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
                v = *reinterpret_cast<const float4*>(&A[g_row * K + g_col]);
            }
            else
            {
                // scalar fallback
                if (g_col + 0 < K) v.x = A[g_row * K + g_col + 0];
                if (g_col + 1 < K) v.y = A[g_row * K + g_col + 1];
                if (g_col + 2 < K) v.z = A[g_row * K + g_col + 2];
                if (g_col + 3 < K) v.w = A[g_row * K + g_col + 3];
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

    // Load B matrix
    // BLOCK_TILE_N == 128 is a multiple of 8
    // Again two float4 loads
    #pragma unroll
    for (int i = 0; i < B_LOADS; i += 4)
    {
        const int lin0  = tid * B_LOADS + i;
        const int s_row = lin0 / BLOCK_TILE_N;
        const int s_col = lin0 % BLOCK_TILE_N;

        const int g_row = k_tile + s_row;
        const int g_col = j0 + s_col;

        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);

        if (g_row < K)
        {
            if (g_col + 3 < N)
            {
                v = *reinterpret_cast<const float4*>(&B[g_row * N + g_col]);
            }
            else
            {
                // scalar fallback
                if (g_col + 0 < N) v.x = B[g_row * N + g_col + 0];
                if (g_col + 1 < N) v.y = B[g_row * N + g_col + 1];
                if (g_col + 2 < N) v.z = B[g_row * N + g_col + 2];
                if (g_col + 3 < N) v.w = B[g_row * N + g_col + 3];
            }
        }

        // Write B shared data with vectorised half2 write
        __half2* B_half2_ptr = reinterpret_cast<__half2*>(&ABdata[MATRIX_B][buffer][s_row][s_col]);
        B_half2_ptr[0] = __float22half2_rn(make_float2(v.x, v.y));
        B_half2_ptr[1] = __float22half2_rn(make_float2(v.z, v.w));
    }
}

__device__ __forceinline__ void Load_Tile(const float* __restrict__ A, const float* __restrict__ B,
                                           __half ABdata[2][2][BLOCK_TILE_K][BLOCK_TILE_M +8],
                                           int M, int N, int K,
                                           int i0,int j0,int tid,
                                           int tile,int buffer)
{

    const int k_tile = tile * BLOCK_TILE_K;

      // Load BLOCK_TILE_M×BLOCK_TILE_K slice of A using col offset tile*BLOCK_TILE_K
        #pragma unroll
        for (int i = 0; i < A_LOADS; i++) {
            //linearised indexing
            const int lin   = tid * A_LOADS + i;

            const int s_row = lin / BLOCK_TILE_K;
            const int s_col = lin % BLOCK_TILE_K;

            const int i_row = i0 + s_row;
            const int j_col = k_tile + s_col;

            // convert to fp16
            ABdata[MATRIX_A][buffer][s_col][s_row] = (i_row < M && j_col < K) ?
                                           __float2half(A[i_row * K + j_col]) : __float2half(0.0f);
        }


    // Load BLOCK_TILE_K×BLOCK_TILE_N slice of B using row offset tile*BLOCK_TILE_K
        #pragma unroll
        for (int i = 0; i < B_LOADS; i++) {
            //linearised indexing
            const int lin   = tid * B_LOADS + i;

            const int s_row = lin / BLOCK_TILE_N;
            const int s_col = lin % BLOCK_TILE_N;

            const int i_row = k_tile + s_row;
            const int j_col = j0 + s_col;

            // convert to fp16 and round to nearest
            ABdata[MATRIX_B][buffer][s_row][s_col] = (i_row < K && j_col < N) ?
                                          __float2half(B[i_row * N + j_col]) : __float2half(0.0f);
        }

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
__device__ __forceinline__ void Write_Tile(float* C, AccFrag &Accumulated_fragments,
                                           int M, int N,
                                           int i0, int j0,
                                           int warp_row, int warp_col)
{
    #pragma unroll
    for (int tile_row_idx = 0; tile_row_idx < NUM_WMMA_TILES_M; tile_row_idx++)
    {
        int row = i0 + warp_row*WARP_TILE_M + tile_row_idx*WMMA_TILE_M;

        #pragma unroll
        for (int tile_col_idx = 0; tile_col_idx < NUM_WMMA_TILES_N; tile_col_idx++)
        {

            int col = j0 + warp_col*WARP_TILE_N + tile_col_idx*WMMA_TILE_N;

            const int row_end = row + (WMMA_TILE_M -1);
            const int col_end = col + (WMMA_TILE_N -1);

            // Need to catch whether the full tile lays within the matrix C
            if (row_end < M && col_end < N) // full tiles written back vectorised
            {
                float* Cptr = C + row * N + col;

                wmma::store_matrix_sync(Cptr,
                                        Accumulated_fragments[tile_row_idx][tile_col_idx],
                                        N,
                                        wmma::mem_row_major);
            }else // Partial tile needs to be written back in a scalar fashion
            {
                for (int i = 0; i < WMMA_TILE_M; i++)
                {
                    int row_s = row + i;
                    for (int j = 0; j < WMMA_TILE_N; j++)
                    {

                        int col_s = col + j;

                        if (row_s < M && col_s < N)
                            C[row_s * N + col_s] = Accumulated_fragments[tile_row_idx][tile_col_idx].x[i * WMMA_TILE_N + j];

                    }
                }
            }
        }
    }
}

__device__ __forceinline__ float ReLU(float z)
{
    return ( z > 0.0f)? z : 0.0f;
}

template<typename AccFrag>
__device__ __forceinline__ void Write_Activation_Tile_opt(float* __restrict__ C,
                                             const float* __restrict__ Bias,
                                             float* __restrict__ D,
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
    float* __restrict__ C_Tile_Warp_Ptr = C_Tile + warp_id * WMMA_TILE_M * SMEM_STRIDE;

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
            const int g_index = row*N + col;


            // Store fragment into padded shared memory
            wmma::store_matrix_sync(C_Tile_Warp_Ptr,
                                    Accumulated_fragments[tile_row][tile_col],
                                    SMEM_STRIDE,
                                    wmma::mem_row_major);

            // Check the full-tile boundary outside the write loop
            // if true, use float4 vectorised operations
            const bool full_tile = (row + WMMA_TILE_M <= M) && (col + WMMA_TILE_N <= N);


            if (full_tile) // vectorised
            {
                const int ti = sub_tile_index / WMMA_TILE_N;
                const int tj = sub_tile_index % WMMA_TILE_N;
                const int offset = ti*N + tj;

                const float bias_val = Bias[row + ti];

                #pragma unroll
                for (int n=0; n < 2; n++)
                {
                    const float4 acc = *reinterpret_cast<const float4*>(&C_Tile_Warp_Ptr[ti*SMEM_STRIDE + tj + 4*n]);
                    const float z0 = acc.x + bias_val;
                    const float z1 = acc.y + bias_val;
                    const float z2 = acc.z + bias_val;
                    const float z3 = acc.w + bias_val;

                    *reinterpret_cast<float4*>(&C[g_index +offset + 4*n]) = make_float4(z0,z1,z2,z3);
                    *reinterpret_cast<float4*>(&D[g_index +offset + 4*n]) = make_float4(ReLU(z0), ReLU(z1),
                                                                                        ReLU(z2),ReLU(z3));
                }

            }
            else // Scalar fallback -> each lane computes 8 elements with a vectorised check
            {
                const int ti = sub_tile_index / WMMA_TILE_N;
                const int tj = sub_tile_index % WMMA_TILE_N;
                const int offset = ti*N + tj;

                if ( (row + ti) < M)
                {
                    const float bias_val = Bias[row + ti];
                    #pragma unroll
                    for (int n = 0; n < 8; n+=4)
                    {
                     if ((col + tj + n + 3) < N)
                     {
                        const float4 acc = *reinterpret_cast<const float4*>(&C_Tile_Warp_Ptr[ti*SMEM_STRIDE + tj + n]);
                        const float z0 = acc.x + bias_val;
                        const float z1 = acc.y + bias_val;
                        const float z2 = acc.z + bias_val;
                        const float z3 = acc.w + bias_val;
                        *reinterpret_cast<float4*>(&C[g_index +offset + n]) = make_float4(z0,z1,z2,z3);
                        *reinterpret_cast<float4*>(&D[g_index +offset + n]) = make_float4(ReLU(z0), ReLU(z1),
                                                                                          ReLU(z2),ReLU(z3));
                     }
                     else
                     {
                        #pragma unroll
                        for (int p = 0; p < 4; p++)
                        {
                            if ((col + tj + n + p) < N)
                            {
                                const float z = C_Tile_Warp_Ptr[ti*SMEM_STRIDE + tj + n + p] + bias_val;
                                C[g_index +offset + n + p] = z;
                                D[g_index +offset + n + p] = ReLU(z);
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
__device__ __forceinline__ void Write_Activation_Tile(float* __restrict__ C,
                                                      const float* __restrict__ Bias,
                                                      float* __restrict__ D,
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
    float* __restrict__ C_Tile_Warp_Ptr = C_Tile + warp_id * WMMA_TILE_M * SMEM_STRIDE;

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
            wmma::store_matrix_sync(C_Tile_Warp_Ptr,
                                    Accumulated_fragments[tile_row][tile_col],
                                    SMEM_STRIDE,
                                    wmma::mem_row_major);

            //
            // if true, use float4 vectorised operations
            const bool full_tile = (row + WMMA_TILE_M <= M) && (col + WMMA_TILE_N <= N);

            float bias_vals[16] = {0.0f};

                // vectorised load of bias values
                #pragma unroll
                for (int bk =0; bk < WMMA_TILE_M; bk+=4)
                {
                    if ((bk + 3) < M)
                    {
                    *reinterpret_cast<float4*>(&bias_vals[bk]) = *reinterpret_cast<const float4*>(&Bias[row + bk]);
                    }
                    else //scalar fallback
                    {
                    #pragma unroll
                    for (int n = 0; n < 4; n++){
                        if (bk+n < M) bias_vals[bk+n] = Bias[row + bk+n];}
                    }
                }


            #pragma unroll
            for (int ti = 0; ti < WMMA_TILE_M; ti++)
            {
                const int g_row = row + ti;
                const float  bias_val  = bias_vals[ti];
                float* __restrict__ C_row = C + g_row * N;
                float* __restrict__ D_row = D + g_row * N;
                const float* C_Tile_row = C_Tile_Warp_Ptr + ti * SMEM_STRIDE;

                // If a full tile, use vectorised access
                if (full_tile)
                {
                    #pragma unroll
                    for (int tj = 0; tj < WMMA_TILE_N; tj += 4)
                    {
                        const float4 acc = *reinterpret_cast<const float4*>(C_Tile_row + tj);

                        const float z0 = acc.x + bias_val;
                        const float z1 = acc.y + bias_val;
                        const float z2 = acc.z + bias_val;
                        const float z3 = acc.w + bias_val;

                        *reinterpret_cast<float4*>(C_row + col + tj) = make_float4(z0,z1,z2,z3);
                        *reinterpret_cast<float4*>(D_row + col + tj) = make_float4(
                                                                       ReLU(z0),
                                                                       ReLU(z1),
                                                                       ReLU(z2),
                                                                       ReLU(z3));
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
                                const float z = C_Tile_row[tj] + bias_val;
                                C_row[g_col]  = z;
                                D_row[g_col]  = ReLU(z);
                            }
                        }
                    }
                }
            }
        }
    }
}



