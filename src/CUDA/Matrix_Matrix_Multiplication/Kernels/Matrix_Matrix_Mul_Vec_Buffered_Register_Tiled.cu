#define TILE 32
#define THREAD_TILE 4

__device__ __forceinline__ int s_ind_A(int si,int sj,int ni,int nj);
__device__ __forceinline__ int s_ind_B(int si,int sj,int ni, int nj);
__device__ __forceinline__ int buf_ind(int buffer,int shared_index);
__device__ __forceinline__ void Load_Tile(float* A,float* B,
                                          float* Adata, float* bdata,
                                          int tile, int buffer,
                                          int row, int col,
                                          int si, int sj,
                                          int M, int N, int K);
__device__ __forceinline__ void Compute_Tile(float* Adata,float* Bdata,int buffer,
                                             float registers[THREAD_TILE][THREAD_TILE],
                                             int si,int sj);

extern "C" __global__ void matrix_matrix_mul_vec_buffered_register_tiled(float* __restrict__ A,
                                                                     float* __restrict__ B,
                                                                     float* __restrict__ C,
                                                                     const int M, const int N, const int K)
{
    // Shared buffered tiled loading of matrix A and B,
    // with accumulation of A*B in the registers of size [4][4]

    // Buffering swaps the shared memory being read and computed, such that both compute
    // and loading overlap.


    extern __shared__ float sdata[];

    float* Adata = sdata;
    // Padded outer shared memory dimension to remove bank conflicts
    float* Bdata = sdata + 2*TILE*(TILE+1);

    float __align__(16) registers[THREAD_TILE][THREAD_TILE] = {0.0f};


    const int sj = threadIdx.x;
    const int si = threadIdx.y;

    // stepping in dims x and y in terms of size of thread tile
    const int row = blockIdx.y*TILE + si*THREAD_TILE;
    const int col = blockIdx.x*TILE + sj*THREAD_TILE;

    // Pre-fetch the first tile
    int buffer = 0;
    Load_Tile(A, B, Adata, Bdata, 0, buffer, row, col, si, sj, M, N, K);

    __syncthreads();

    for (int tile = 0; tile < K; tile += TILE)
    {
        int next_buffer = buffer ^ 1;

        // load the next tile
        if (tile + TILE < K)
            {
                Load_Tile(A, B, Adata, Bdata, tile + TILE, next_buffer, row, col, si, sj, M, N, K);
            }

        // compute on using previously loaded tile
        Compute_Tile(Adata, Bdata, buffer, registers, si, sj);

        __syncthreads();

        // swap the buffers
        buffer = next_buffer;
    }

    #pragma unroll
    // write back 16 values to global memory
    for (int i = 0;i < THREAD_TILE; i++)
    {
        #pragma unroll
        for (int j = 0;j < THREAD_TILE; j++)
            {
                int c_row = row + i;
                int c_col = col + j;

                if (c_row < M && c_col < N)
                    {
                        C[c_row*N + c_col] = registers[i][j];
                    }
            }
    }
}

__device__ __forceinline__ void Load_Tile(float* A,float* B,
                          float* Adata, float* Bdata,
                          int tile, int buffer,
                          int row, int col,
                          int si, int sj,
                          int M, int N, int K)
{

    // populate the buffer (0,1)
    // K and N are multiples of 4
        #pragma unroll
        for (int ni = 0; ni < THREAD_TILE; ni++)
        {

                int ai = row + ni;
                int aj = tile + sj*THREAD_TILE;

                int bi = tile + si*THREAD_TILE + ni;
                int bj = col;

                if (ai < M)
                {
                    if (aj + 3 < K)
                    {
                        const float4 vec4A = reinterpret_cast<const float4*>(A +ai*K + aj)[0];
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,0))] = vec4A.x;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,1))] = vec4A.y;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,2))] = vec4A.z;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,3))] = vec4A.w;
                    }
                    else
                    {
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,0))] = (aj < K)? A[ai*K + aj] : 0.0f;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,1))] = (aj +1 < K)? A[ai*K + aj+1] : 0.0f;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,2))] = (aj +2 < K)? A[ai*K + aj+2] : 0.0f;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,3))] = 0.0f;
                    }
                }
                else
                {
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,0))] = 0.0f;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,1))] = 0.0f;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,2))] = 0.0f;
                        Adata[buf_ind(buffer,s_ind_A(si,sj,ni,3))] = 0.0f;
                }


                if ( bi < K)
                {
                    if ( bj + 3 < N)
                    {
                        const float4 vec4B = reinterpret_cast<const float4*>(B+bi*N + bj)[0];
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,0))] = vec4B.x;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,1))] = vec4B.y;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,2))] = vec4B.z;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,3))] = vec4B.w;
                    }
                    else
                    {
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,0))] = (bj < N)? B[bi*N + bj] : 0.0f;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,1))] = (bj +1 < N)? B[bi*N + bj+1] : 0.0f;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,2))] = (bj +2 < N)? B[bi*N + bj+2] : 0.0f;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,3))] = 0.0f;
                    }
                }
                else
                {
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,0))] = 0.0f;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,1))] = 0.0f;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,2))] = 0.0f;
                        Bdata[buf_ind(buffer,s_ind_B(si,sj,ni,3))] = 0.0f;
                }


        }

}

__device__ __forceinline__ void Compute_Tile(float* Adata,float* Bdata,int buffer,
                                             float registers[THREAD_TILE][THREAD_TILE],
                                             int si,int sj)
{
     float a[THREAD_TILE];
     float b[THREAD_TILE];

         // accumulate in registers, where the k dimension is contiguous in shared memory
        #pragma unroll
        for (int k = 0; k < TILE; k++)
        {
                #pragma unroll
                for (int n = 0; n < THREAD_TILE; n++)
                {
                   a[n] = Adata[buf_ind(buffer,s_ind_A(si,0,n,k))];
                   b[n] = Bdata[buf_ind(buffer,s_ind_B(0,sj,k,n))];
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
}

__device__ __forceinline__ int s_ind_A(int si,int sj,int ni,int nj)
{
    return (ni + si * THREAD_TILE)*(TILE + 1) + sj*THREAD_TILE + nj;
}

__device__ __forceinline__ int s_ind_B(int si,int sj,int ni, int nj)
{
    return (nj + sj * THREAD_TILE)*(TILE + 1) + si*THREAD_TILE + ni;
}

__device__ __forceinline__ int buf_ind(int buffer,int shared_index)
{
return buffer*TILE*(TILE + 1) + shared_index;
}