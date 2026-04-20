#include <cuda_runtime.h>

// ------------Setup----------------
// block tile
// [TILE M][TILE K] -> A
// [TILE K][TILE N] -> B
constexpr int BLOCK_TILE_M  = 128;  // block tile rows
constexpr int BLOCK_TILE_N  = 128;  // block tile cols
constexpr int BLOCK_TILE_K  = 16;   // block tile depth (K dimension)

// warp tile
constexpr int WARP_TILE_M   = 64;   // warp tile rows
constexpr int WARP_TILE_N   = 32;   // warp tile cols

// thread tile
constexpr int THREAD_TILE_M = 8;    // per-thread register rows
constexpr int THREAD_TILE_N = 8;    // per-thread register cols

constexpr int THREAD_TILE_VEC_M = THREAD_TILE_M/4;    // per-thread register rows
constexpr int THREAD_TILE_VEC_N = THREAD_TILE_N/4;    // per-thread register cols


// Number of warps in each block dimension
constexpr int WARPS_M       = BLOCK_TILE_M / WARP_TILE_M; // 2
constexpr int WARPS_N       = BLOCK_TILE_N / WARP_TILE_N; // 4
constexpr int NUM_WARPS     = WARPS_M * WARPS_N;          // 8
constexpr int BLOCK_THREADS = NUM_WARPS * 32;             // 256

// Thread mapping within a warp
// Warps cover the warp tile WARP_TILE_M * WARP_TILE_N
// WARP_THREADS_M * WARP_THREADS_N must equal 32 (warp size)
constexpr int WARP_THREADS_M = WARP_TILE_M / THREAD_TILE_M; // 8  threads rows
constexpr int WARP_THREADS_N = WARP_TILE_N / THREAD_TILE_N; // 4  threads cols

// How many elements each thread reads into shared memory per K tile
constexpr int A_LOADS = (BLOCK_TILE_M * BLOCK_TILE_K) / BLOCK_THREADS; // 8
constexpr int B_LOADS = (BLOCK_TILE_K * BLOCK_TILE_N) / BLOCK_THREADS; // 8
// ----------------------------------


__device__ __forceinline__ void Load_Tile( const float* A,const float* B,
                                           float Adata[2][BLOCK_TILE_K][BLOCK_TILE_M +4],
                                           float Bdata[2][BLOCK_TILE_K][BLOCK_TILE_N +4],
                                           int M, int N, int K,
                                           int i0,int j0,int tid,
                                           int tile,int buffer);

__device__ __forceinline__ void Compute_Tile(float Adata[2][BLOCK_TILE_K][BLOCK_TILE_M +4],
                                             float Bdata[2][BLOCK_TILE_K][BLOCK_TILE_N +4],
                                             float registers[THREAD_TILE_M][THREAD_TILE_N],
                                             float A_row_slice[THREAD_TILE_M],
                                             float B_column_slice[THREAD_TILE_N],
                                             int warp_row, int warp_col,
                                             int thr_row, int thr_col,
                                             int buffer);

__device__ __forceinline__ void Write_Activation_Tile(float* C, const float* b, float* D,
                                                      float registers[THREAD_TILE_M][THREAD_TILE_N],
                                                      int M, int N,
                                                      int i0, int j0,
                                                      int warp_row, int warp_col,
                                                      int thr_row, int thr_col);

__device__ __forceinline__ float ReLU(float z);

extern "C" __global__ void forward_propagate_hidden_layer_kernel(const float* __restrict__ A,
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

    // row and column indices of a thread within the warp tile
    // lane = (WARP_THREADS_N * row) + col         (row-major)
    const int thr_row    = lane / WARP_THREADS_N;
    const int thr_col    = lane % WARP_THREADS_N;

    // Global starting index of this block's output tile -> C[i0 +i,j0 + j]
    const int i0 = blockIdx.y * BLOCK_TILE_M;
    const int j0 = blockIdx.x * BLOCK_TILE_N;

    // pad the number of columns to suppress shared memory bank conflicts
    __shared__ __align__(16) float Adata[2][BLOCK_TILE_K][BLOCK_TILE_M +4];
    __shared__ __align__(16) float Bdata[2][BLOCK_TILE_K][BLOCK_TILE_N +4];

    float registers[THREAD_TILE_M][THREAD_TILE_N] = {};   // accumulate into registers
    float A_row_slice[THREAD_TILE_M];          // A column-slice for one k step
    float B_column_slice[THREAD_TILE_N];          // B row-slice   for one k step


    const int NUMBER_OF_K_TILES = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

    int buffer = 0;
    //pre-load into buffer 0
    Load_Tile(A, B, Adata, Bdata, M, N, K, i0, j0, tid, 0, buffer);

    __syncthreads();

    for (int tile = 0; tile < NUMBER_OF_K_TILES; tile++) {
        // switch the buffers
        buffer = tile & 1;      // current buffer for compute
        const int next_buffer = buffer ^ 1;    // next buffer to load, toggles 1,0,...

        // load the next tile into the other buffer
        if (tile + 1 < NUMBER_OF_K_TILES)
            Load_Tile(A, B, Adata, Bdata, M, N, K, i0, j0, tid, tile + 1, next_buffer);

        // compute on the previously loaded tile
        Compute_Tile(Adata, Bdata, registers,
                     A_row_slice, B_column_slice,
                     warp_row, warp_col, thr_row, thr_col,
                     buffer);

        // ensure the next tile has loaded into the buffer
        __syncthreads();
    }

    Write_Activation_Tile(C, b, D, registers, M, N, i0, j0, warp_row, warp_col, thr_row, thr_col);

}

__device__ __forceinline__ void Load_Tile(const float* A,const float* B,
                                          float Adata[2][BLOCK_TILE_K][BLOCK_TILE_M +4],
                                          float Bdata[2][BLOCK_TILE_K][BLOCK_TILE_N +4],
                                          int M, int N, int K,
                                          int i0,int j0,int tid,
                                          int tile,int buffer)
{

      // Load BLOCK_TILE_M×BLOCK_TILE_K slice of A using col offset tile*BLOCK_TILE_K
        #pragma unroll
        for (int i = 0; i < A_LOADS; i++) {

            const int lin   = tid * A_LOADS + i;
            const int s_row = lin / BLOCK_TILE_K;
            const int s_col = lin % BLOCK_TILE_K;

            const int i_row = i0 + s_row;
            const int j_col = tile * BLOCK_TILE_K + s_col;

            Adata[buffer][s_col][s_row] = (i_row < M && j_col < K) ? A[i_row * K + j_col] : 0.0f;
        }

    // Load BLOCK_TILE_K×BLOCK_TILE_N slice of B using row offset tile*BLOCK_TILE_K
        #pragma unroll
        for (int i = 0; i < B_LOADS; i++) {
            const int lin   = tid * B_LOADS + i;
            const int s_row = lin / BLOCK_TILE_N;
            const int s_col = lin % BLOCK_TILE_N;

            const int i_row = tile * BLOCK_TILE_K + s_row;
            const int j_col = j0 + s_col;

            Bdata[buffer][s_row][s_col] = (i_row < K && j_col < N) ? B[i_row * N + j_col] : 0.0f;
        }

}

__device__ __forceinline__ void Compute_Tile(float Adata[2][BLOCK_TILE_K][BLOCK_TILE_M +4],
                                             float Bdata[2][BLOCK_TILE_K][BLOCK_TILE_N +4],
                                             float registers[THREAD_TILE_M][THREAD_TILE_N],
                                             float A_row_slice[THREAD_TILE_M],
                                             float B_column_slice[THREAD_TILE_N],
                                             int warp_row, int warp_col,
                                             int thr_row, int thr_col,
                                             int buffer)
{

   #pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; k++) {
            // Vectorised THREAD_TILE_M consecutive rows from shared memory
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_VEC_M; m++) {

                const int row = warp_row * WARP_TILE_M + thr_row * THREAD_TILE_M + 4*m;
                const float4 vec4A = reinterpret_cast<const float4*>(&Adata[buffer][k][row])[0];
                reinterpret_cast<float4*>(&A_row_slice[4*m])[0] = vec4A;

            }

            // Vectorised THREAD_TILE_N consecutive cols from shared memory
            #pragma unroll
            for (int n = 0; n < THREAD_TILE_VEC_N; n++) {
                const int col = warp_col * WARP_TILE_N + thr_col * THREAD_TILE_N + 4*n;
                const float4 vec4B = reinterpret_cast<const float4*>(&Bdata[buffer][k][col])[0];
                reinterpret_cast<float4*>(&B_column_slice[4*n])[0] = vec4B;
            }

            // FMA accumulation in registers
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++)
                #pragma unroll
                for (int n = 0; n < THREAD_TILE_N; n++)
                    registers[m][n] += A_row_slice[m] * B_column_slice[n];
        }


}

__device__ __forceinline__ float ReLU(float z)
{
    return ( z > 0.0f)? z : 0.0f;
}

__device__ __forceinline__ void Write_Activation_Tile(float* C, const float* b, float* D,
                                                      float registers[THREAD_TILE_M][THREAD_TILE_N],
                                                      int M, int N,
                                                      int i0, int j0,
                                                      int warp_row, int warp_col,
                                                      int thr_row, int thr_col)
{

  // write back to global memory THREAD_TILE_M*THREAD_TILE_N elements
    #pragma unroll
    for (int m = 0; m < THREAD_TILE_M; m++)
    {
        const int row = i0 + warp_row * WARP_TILE_M + thr_row * THREAD_TILE_M + m;

        #pragma unroll
        for (int n = 0; n < THREAD_TILE_N; n++)
        {
            const int col = j0 + warp_col * WARP_TILE_N + thr_col * THREAD_TILE_N + n;

            if (row < M && col < N)
            {
                const float z = registers[m][n] + b[row];

                C[row * N + col] = z;
                D[row * N + col] = ReLU(z);
            }
        }
    }

}