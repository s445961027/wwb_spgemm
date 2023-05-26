#include<cuda_runtime.h>
#include<stdio.h>

//不分块naive
__global__ void MatrixMul_naive(const float *A,const float *B,float *C,const int M,const int K,const int N)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0;
#pragma unroll
    for( int k = 0; k < K; k++){
        sum += A[y*K+k] * B[k*N+x]; 
    }
    C[y*N+x] = sum;

}


//分块，利用sharememory，每个thread计算一个element of C,每个块用内积的方式计算,非正方阵
template<int TILE_SIZE_M,int TILE_SIZE_K,int TILE_SIZE_N>
__global__ void MatrixMul1(const float *A,const float *B,float *C,const int M,const int K,const int N)
{

    __shared__ float local_A[TILE_SIZE_M][TILE_SIZE_K];
    __shared__ float local_B[TILE_SIZE_K][TILE_SIZE_N];

    // int baseY = TILE_SIZE_M * blockIdx.y;
    // int baseX = TILE_SIZE_N * blockIdx.x;
    float sum = 0.0;
    for(int tileid = 0; tileid < K/TILE_SIZE_K; tileid++)
    {
        //sharedmemory存在重复读取，太笨了，且存在bank冲突
        
        
        if(threadIdx.x < TILE_SIZE_K ){
            local_A[threadIdx.y][threadIdx.x] = A[(blockDim.y * blockIdx.y + threadIdx.y) * K + tileid * TILE_SIZE_K + threadIdx.x];
        }
        if(threadIdx.y < TILE_SIZE_K){
            local_B[threadIdx.y][threadIdx.x] = B[(tileid * TILE_SIZE_K + threadIdx.y) * N + blockDim.x * blockIdx.x + threadIdx.x];
        }
        __syncthreads();
    #pragma unroll
        for(int k = 0; k < TILE_SIZE_K; k++){
            sum += local_A[threadIdx.y][k] * local_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[N * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x] = sum;
     



}

//分块，每个thread计算多个element of C,向量内积方式  每个线程负责4 * 4个element of C，A100 每个SM有128个FFMA，4 * 4跑不满算力
template<int TILE_SIZE_M,int TILE_SIZE_K,int TILE_SIZE_N,int THREAD_BLOCK_M, int THREAD_BLOCK_N>
__global__ void MatrixMul2(const float *A,const float *B,float *C,const int M,const int K,const int n)
{
    __shared__ float local_A[TILE_SIZE_M][TILE_SIZE_K];
    __shared__ float local_B[TILE_SIZE_K][TILE_SIZE_N];
    float local_C[THREAD_BLOCK_M][THREAD_BLOCK_N];
    //在每个tile内每个线程负责的小块的起始位置
    
    int In_Tile_startX = threadIdx.x * 4;
    int In_Tile_startY = threadIdx.y * 4;
    
    for(int tileid = 0; tileid < K/TILE_SIZE_K; tileid++)
    {   
        //读取数据到共享内存 
        if(In_Tile_startX < TILE_SIZE_K){
            int index_A_Y = blockDim.y * THREAD_BLOCK_M * blockIdx.y + threadIdx.y * THREAD_BLOCK_M;
            int index_A_X = tileid * TILE_SIZE_K + threadIdx.x * THREAD_BLOCK_N;
            for(int j = 0; j < 4; j++)
            {   
                //

                // int index_A = (blockDim.y * THREAD_BLOCK_M * blockIdx.y + threadIdx.y * THREAD_BLOCK_M + j) * K + tileid * TILE_SIZE_K + threadIdx.x * 4;
                  
                local_A[In_Tile_startY + j][In_Tile_startX] = A[(index_A_Y + j) * K + index_A_X];
                local_A[In_Tile_startY + j][In_Tile_startX + 1] = A[(index_A_Y + j) * K + index_A_X + 1];
                local_A[In_Tile_startY + j][In_Tile_startX + 2] = A[(index_A_Y + j) * K + index_A_X + 2];
                local_A[In_Tile_startY + j][In_Tile_startX + 3] = A[(index_A_Y + j) * K + index_A_X + 3];
            }
        }
        if(In_Tile_startY < TILE_SIZE_K)
        {   
            int index_B_Y = (tileid * TILE_SIZE_K + threadIdx.y * THREAD_BLOCK_M);
            int index_B_X = (blockDim.x * THREAD_BLOCK_N * blockIdx.x + threadIdx.x * THREAD_BLOCK_N);

            for(int i = 0; i < 4; i++)
            {
                
                local_B[In_Tile_startY + j][In_Tile_startX] = B[(index_B_Y + j) * N + index_B_X];
                local_B[In_Tile_startY + j][In_Tile_startX + 1] = B[(index_B_Y + j) * N + index_B_X + 1];
                local_B[In_Tile_startY + j][In_Tile_startX + 2] = B[(index_B_Y + j) * N + index_B_X + 2];
                local_B[In_Tile_startY + j][In_Tile_startX + 3] = B[(index_B_Y + j) * N + index_B_X + 3];
            }
        }   
        __syncthreads();
        
        for(int k = 0; k < TILE_SIZE_K; k++)
        {
            for(int m = 0; m < THREAD_BLOCK_M; m++)
            {
                for(int n = 0; n < THREAD_BLOCK_N; n++)
                {
                    //外积   
                    local_C[m][n] += local_A[threadIdx.y * THREAD_BLOCK_M + m][k]*local_B[k][threadIdx.x * THREAD_BLOCK_N + n];
                }
            }
        }
        __syncthreads();
    }
    //结果写回主存,可以做float4向量化写回
    int y = (blockDim.y * blockIdx.y + threadIdx.y) * THREAD_BLOCK_M;
    int x = (blockDim.x * blockIdx.x + threadIdx.x) * THREAD_BLOCK_N;
    for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++)
    {
        for(int thread_x = 0; thread_x < THREAD_BLOCK_N; thread_x++)
        {
            C[(y+thread_y) * N + x + thread_x] = local_C[thread_y][thread_x];
        }
    }

    
}


//分块，每个thread计算多个element，向量外积方式
template<int TILE_SIZE_M,int TILE_SIZE_K,int TILE_SIZE_N>
__global__ void MatrixMul3(){
    


}

//