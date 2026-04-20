#define TILE 64
#define WARP_TILE 32
#define THREAD_TILE 8

__device__ __forceinline__ int s_ind_A(int si,int sj,int ni,int nj);
__device__ __forceinline__ int s_ind_B(int si,int sj,int ni, int nj);

extern "C" __global__ void matrix_matrix_mul_warp_register_tiled( const float* __restrict__ A,
                                                             const float* __restrict__ B,
                                                             float* __restrict__ C,
                                                             const int M, const int N, const int K)
{
    // Shared tiled loading of matrix A and B,
    // with accumulation of A*B in the registers of size [8][8]

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp = tid >> 5;

    const int WARPS_X = TILE/WARP_TILE;
    int warp_row = warp / WARPS_X;
    int warp_col = warp % WARPS_X;

    extern __shared__ float sdata[];

    float* Adata = &sdata[0];
    float* Bdata = &sdata[0] + TILE*TILE;

    float registers[THREAD_TILE][THREAD_TILE];

     #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++)
            {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++)
                {
                    registers[i][j] = 0.0f;
                }
            }

    const int sj = threadIdx.x;
    const int si = threadIdx.y;

    const int row = blockIdx.y*TILE + warp_row*WARP_TILE + THREAD_TILE*si;
    const int col = blockIdx.x*TILE + warp_col*WARP_TILE + THREAD_TILE*sj;


    for (int tile = 0; tile < K; tile += TILE)
    {
        #pragma unroll
        for (int ni = 0; ni < THREAD_TILE; ni++)
        {
        #pragma unroll
            for (int nj = 0; nj < THREAD_TILE; nj++)
            {
                int ai = row + ni;
                int aj = tile + sj*THREAD_TILE + nj;

                int bi = tile + si*THREAD_TILE + ni;
                int bj = col + nj;

                Adata[s_ind_A(si,sj,ni,nj)] = (ai < M && aj < K)? A[ai*K + aj] : 0.0f;
                Bdata[s_ind_B(si,sj,ni,nj)] = (bi < K && bj < N)? B[bi*N + bj] : 0.0f;
            }

        }

        __syncthreads();

         // accumulate in registers, where the k dimension is contiguous in shared memory
        #pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            float a = Adata[(si*THREAD_TILE)*TILE + k];
            float b = Bdata[(sj*THREAD_TILE)*TILE + k];

            #pragma unroll
            for (int src = 0; src < 32; src++)
            {
                float a_sh = __shfl_sync(0xffffffff,a,src);
                float b_sh = __shfl_sync(0xffffffff,b,src);

                 #pragma unroll
                for (int i = 0; i < THREAD_TILE; i++)
                {
                    #pragma unroll
                    for (int j = 0; j < THREAD_TILE; j++)
                    {
                    registers[i][j] += a_sh * b_sh;
                    }
                }
            }

        }

        __syncthreads();
    }

    // write back 64 values to global memory
    #pragma unroll
    for (int i = 0;i < THREAD_TILE; i++)
    {
        #pragma unroll
        for (int j = 0;j < THREAD_TILE; j++)
            {
                int c_row = row + i;
                int c_col = col + j;

                if (c_row < M && c_col < N)
                {C[c_row*N + c_col] = registers[i][j];}
            }
    }

}

__device__ __forceinline__ int s_ind_A(int si,int sj,int ni,int nj)
{
    return (ni + si * THREAD_TILE)*TILE + sj*THREAD_TILE + nj;
}

__device__ __forceinline__ int s_ind_B(int si,int sj,int ni, int nj)
{
    return (nj + sj * THREAD_TILE)*TILE + si*THREAD_TILE + ni;
}