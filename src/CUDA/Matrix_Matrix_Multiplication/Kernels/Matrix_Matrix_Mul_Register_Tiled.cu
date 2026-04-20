#define TILE 64
#define THREAD_TILE 8

__device__ __forceinline__ int s_ind_A(int si,int sj,int ni,int nj);
__device__ __forceinline__ int s_ind_B(int si,int sj,int ni, int nj);

extern "C" __global__ void matrix_matrix_mul_register_tiled( const float* __restrict__ A,
                                                             const float* __restrict__ B,
                                                             float* __restrict__ C,
                                                             const int M, const int N, const int K)
{
    // Shared tiled loading of matrix A and B,
    // with accumulation of A*B in the registers of size [8][8]

    extern __shared__ float sdata[];

    float* Adata = &sdata[0];
    // Outer-most dimension padded by 1 to remove bank conflicts
    float* Bdata = &sdata[0] + TILE*(TILE+1);

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

    const int row = blockIdx.y*TILE + THREAD_TILE*si;
    const int col = blockIdx.x*TILE + THREAD_TILE*sj;


    for (int tile = 0; tile < K; tile += TILE)
    {
        #pragma unroll
        for (int ni = 0; ni < THREAD_TILE; ni++)
        {
           int ai = row + ni;
           int bi = tile + si*THREAD_TILE + ni;
        #pragma unroll
            for (int nj = 0; nj < THREAD_TILE; nj++)
            {

                int aj = tile + sj*THREAD_TILE + nj;
                int bj = col + nj;

                Adata[s_ind_A(si,sj,ni,nj)] = (ai < M && aj < K)? A[ai*K + aj] : 0.0f;
                Bdata[s_ind_B(si,sj,ni,nj)] = (bi < K && bj < N)? B[bi*N + bj] : 0.0f;
            }

        }

        __syncthreads();

        float a[THREAD_TILE];
        float b[THREAD_TILE];

         // accumulate in registers, where the k dimension is contiguous in shared memory
        #pragma unroll
        for (int k = 0; k < TILE; k++)
        {
                #pragma unroll
                for (int n = 0; n < THREAD_TILE; n++)
                {
                   int index_A = (n + si*THREAD_TILE)*(TILE+1) + k;
                   int index_B = (n + sj*THREAD_TILE)*(TILE+1) + k;
                   a[n] = Adata[index_A];
                   b[n] = Bdata[index_B];
                }

                #pragma unroll
                for (int i = 0; i < THREAD_TILE; i++)
                {
                    #pragma unroll
                    for (int j = 0; j < THREAD_TILE; j++)
                    {
                    registers[i][j] += a[i]*b[j];
                    }
                }

        }

        __syncthreads();
    }

    // write back 64 values to global memory
    #pragma unroll
    for (int i = 0;i < THREAD_TILE; i++)
    {
        int c_row = row + i;
        #pragma unroll
        for (int j = 0;j < THREAD_TILE; j++)
            {
                int c_col = col + j;

                if (c_row < M && c_col < N)
                {C[c_row*N + c_col] = registers[i][j];}
            }
    }

}

__device__ __forceinline__ int s_ind_A(int si,int sj,int ni,int nj)
{
    return (ni + si * THREAD_TILE)*(TILE+1) + sj*THREAD_TILE + nj;
}

__device__ __forceinline__ int s_ind_B(int si,int sj,int ni, int nj)
{
    return (nj + sj * THREAD_TILE)*(TILE+1) + si*THREAD_TILE + ni;
}



