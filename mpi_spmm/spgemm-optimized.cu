#include <string.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>
#include "cuda.h"
#include "common.h"
#include "utils.h"
#include "mpi.h"
//#define wwb
//#define w_file
const char* version_name = "optimized version";

void MPI_CHECK_ERROR(int status){
	if (status != MPI_SUCCESS) 
	{
    	char error_string[MPI_MAX_ERROR_STRING];
    	int length_of_error_string;
    	MPI_Error_string(status, error_string, &length_of_error_string);
    	fprintf(stderr, "MPI_Isend Error: %s\n", error_string);
    
	}
}


void preprocess(dist_matrix_t *matA, dist_matrix_t *matB) {

}

void destroy_additional_info(void *additional_info) {

}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}


//getRowNnz看懂后 需要修改
void getRowNnz(dist_matrix_t *mat, dist_matrix_t* matB, dist_matrix_t* matC, std::vector<index_t>& c_idx)
{	
	int signal = 1;
	matC->r_pos[0] = 0;
	int *tmp = NULL;
	int k = matB->global_m;
	tmp = new int[k];
	memset(tmp , -1, k * sizeof(int));
	int nnz = 0;
#ifdef wwb
	// printf("myID is %d\n",getpid());
	
	// while(signal){
	// 	sleep(2);
	// }
#endif
	//int pos = 0;
	for(int i = 0; i < matC->global_m; i++){
		for(int j = mat->r_pos[i]; j < mat->r_pos[i + 1]; j++){
			int jj = mat->c_idx[j];
			for(int k = matB->r_pos[jj]; k < matB->r_pos[jj + 1]; k++){
				int kk = matB->c_idx[k];
				if(tmp[kk] != i){
					c_idx.push_back(kk);
					tmp[kk] = i;
					nnz++;
				}
			}
		}
		matC->r_pos[i + 1] = nnz;
	}
#ifdef wwb
	printf("nnz is %d\n",nnz);
#endif
	matC->global_nnz = nnz;
	delete []tmp;
}

__global__ void SortAndRowExc(index_t *dptr_colindex_C, index_t *dptr_rowindex_C, index_t *dptr_offset_C, int m)
{
	int rowindex = threadIdx.x + blockDim.x * blockIdx.x;
	if (rowindex < m)
	{
		// 找到当前行元素再col和val数组的下标范围
		int left = dptr_offset_C[rowindex];
		int right = dptr_offset_C[rowindex + 1];
		// 排序
		int n = right - left;
		int p, q, gap;
		for (gap = n / 2; gap > 0; gap /= 2)
		{
			for (p = 0; p < gap; p++)
			{
				for (q = p + gap + left; q < n + left; q += gap)
				{
					if (dptr_colindex_C[q] < dptr_colindex_C[q - gap])
					{
						int tmp = dptr_colindex_C[q];
						int k = q - gap;
						while (k >= left && dptr_colindex_C[k] > tmp)
						{
							dptr_colindex_C[k + gap] = dptr_colindex_C[k];
							k -= gap;
						}
						dptr_colindex_C[k + gap] = tmp;
					}
				}
			}
		}
		//  初始化行号数组的值
		for (int i = left; i < right; i++)
		{
			dptr_rowindex_C[i] = rowindex;
		}
	}
}


// 核函数每个线程负责结果C中的每个位置 通过此位置对用的行号和列号 去遍历A和B中相应的元素乘积再相加 得到的结果存到当前位置
__global__ void compute(index_t *dptr_rowindex_C, index_t *dptr_colindex_C, data_t *dptr_value_C,
					const index_t *dptr_offset_A, const index_t *dptr_offset_B,
					const index_t *dptr_colindex_A, const index_t *dptr_colindex_B,
					data_t *dptr_value_A, data_t *dptr_value_B,
					int nonzero, int r_start,int r_end, data_t alpha)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x; // 对应在value_C数组的下标
	if (idx < nonzero)
	{
		if (idx != 0 && dptr_colindex_C[idx] == dptr_colindex_C[idx - 1] && dptr_rowindex_C[idx] == dptr_rowindex_C[idx - 1])
		{
			dptr_value_C[idx] = 0.0;
		}
		else
		{

			int row = dptr_rowindex_C[idx]; // 当前位置所对应的行号与列号
			int col = dptr_colindex_C[idx]; // 通过行号确定遍历A的非0元素所在的列号 通过列号确定寻找B的列号为col的元素
			double sum = 0;					// 记录当前位置存入的最终结果
			double value_A;
			double value_B;
			int A_begin = dptr_offset_A[row];
			int A_end = dptr_offset_A[row + 1];//所以matA->rpos要将起始偏移弄成0
			for (int jj = A_begin; jj < A_end; jj++)
			{	
				// jj为当前第row行的非0元素在value_A数组与col_A数组中的起始位置
				value_A = dptr_value_A[jj];	 // 当前A的值
				int j = dptr_colindex_A[jj]; // j为当前A的第row行中非0元素所处于的列号
				
				// 折半查找 寻找B中第j行的列号为col的非0元素 与A位置的元素相乘再相加得到最终结果
				if( j >= r_start && j < r_end)//用于判断当前B的某些行的数据在不在本进程中，若不在直接不用计算。
				{	
					int left = dptr_offset_B[j];//
					int right = dptr_offset_B[j + 1] - 1;
					int mid = 0;
					while (left <= right)
					{	
						
						mid = left + (right - left) / 2;
						if (dptr_colindex_B[mid] < col)
						{
							left = mid + 1;
						}
						else if (dptr_colindex_B[mid] > col)
						{
							right = mid - 1;
						}
						else if (dptr_colindex_B[mid] == col)
						{
							value_B = dptr_value_B[ mid - dptr_offset_B[r_start]];//这里需要偏移一下，需要减一个起始偏移量r_pos[r_start];
							sum = sum + value_A * value_B;
							break;
						}
					}
				}
			}
			dptr_value_C[idx] += sum * alpha; // 最终结果需要乘以一个系数
		}
	}
}

void spgemm(dist_matrix_t *matA, dist_matrix_t *matB, dist_matrix_t *matC ,int p_id, int process_num)
{   
	cudaError_t error;

    std::vector<index_t>c_idx;
    
    //注意，这个C是每个进程独有的,其global_m由当前进程的matA决定
    matC->global_m = matA->global_m;
    matC->r_pos = (int*)malloc(sizeof(int)*(matC->global_m+1));
#ifdef wwb
	printf("process[%d]'s global_m is %d\n",p_id,matC->global_m);
#endif
    //需要所有线程都包含整个matB的所有r_pos和c_id,但只包含部分的matB->values
    //需要记录matB->values的上下限，超过上下限就不计算
    //用于计算当前进程的C的nnz、r_pos和c_id
    getRowNnz(matA, matB, matC, c_idx);
    matC->values = (data_t *)malloc(matC->global_nnz * sizeof(data_t));
	if(matC->values == NULL)printf("matC.values malloc error\n");
	matC->c_idx = (index_t *)malloc(matC->global_nnz * sizeof(index_t));
#ifdef wwb
	printf("after getRownnz\n");
#endif
    CUDACHECK(cudaMalloc(&(matC->gpu_r_pos), sizeof(data_t)*(matC->global_m+1) ));
	CUDACHECK(cudaMalloc(&(matC->gpu_c_idx), (matC->global_nnz) * sizeof(index_t)));
	CUDACHECK(cudaMalloc(&(matC->gpu_values), (matC->global_nnz) * sizeof(data_t)));
	CUDACHECK(cudaMemcpy(matC->gpu_r_pos, matC->r_pos, sizeof(index_t)*(matC->global_m + 1), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(matC->gpu_c_idx,&c_idx[0], (matC->global_nnz) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(matC->gpu_values, 0, matC->global_nnz * sizeof(data_t)));
	//辅助行号数组，用于存储当前C的values来自于A的哪一行
    index_t *C_rowIndex;
	CUDACHECK(cudaMalloc(&C_rowIndex,matC->global_nnz * sizeof(index_t)));

    dim3 dimBlock(256);
	dim3 dimGrid((matC->global_m + dimBlock.x - 1) / dimBlock.x);
	SortAndRowExc<<<dimGrid, dimBlock>>>( matC->gpu_c_idx, C_rowIndex, matC->gpu_r_pos, matC->global_m);
	dim3 dimGrids((matC->global_nnz + dimBlock.x - 1) / dimBlock.x);


#ifdef wwb
	index_t * h_C_rowIndex = (index_t *)malloc(sizeof(index_t) * matC->global_nnz);
	CUDACHECK(cudaMemcpy(h_C_rowIndex, C_rowIndex, matC->global_nnz * sizeof(index_t), cudaMemcpyDeviceToHost));

#endif
	error = cudaGetLastError();
	CUDACHECK(error);
#ifdef wwb
	//cudaMemcpy(matC->c_idx, matC->gpu_c_idx, matC->global_nnz * sizeof(index_t), cudaMemcpyDeviceToHost);
	// //cudaDeviceSynchronize();
	// printf("my p_id is %d\n",getpid());
	// int signal = 1;
	// while(signal){
	// 	sleep(2);
	// }
#endif

	//start compute
    int prior_pid, next_pid;
    int send_val_num, recv_val_num;
    int r_st, r_ed;
	additional_info * info = (additional_info *)matB->additional_info;
    //下一个进程号和前一个进程号
    prior_pid = p_id == 0 ? process_num - 1 : p_id - 1;
    next_pid  = ( p_id + 1 ) % process_num;
    //用于交换buffer的指针
    int status;
    data_t *tmp_data_swap;
    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);
#ifdef wwb
	printf("before loop\n");
#endif
    for(int i = 0; i < process_num; i++)
    {   
        MPI_Request  send_req1, send_req2, send_req3;
        MPI_Request  recv_req1, recv_req2, recv_req3;
        
        r_st = info->r_start; 
        r_ed = info->r_end;
         
        //传输matB的st,传输给下个进程，
        status = MPI_Isend(&(info->r_start),1, MPI_INT, next_pid, 1, MPI_COMM_WORLD, &send_req1);
		MPI_CHECK_ERROR(status);
        //传输matB的ed
        status = MPI_Isend(&(info->r_end),  1, MPI_INT, next_pid, 1, MPI_COMM_WORLD, &send_req2);
		MPI_CHECK_ERROR(status);
        
        send_val_num = matB->r_pos[r_ed] - matB->r_pos[r_st];
        
        
        //r_start和r_end的recv,利用这些数据将recv_val_num计算出来
        status = MPI_Irecv(&(info->r_start_buffer), 1, MPI_INT, prior_pid, 1, MPI_COMM_WORLD, &recv_req1);
		MPI_CHECK_ERROR(status);
        status = MPI_Irecv(&(info->r_end_buffer),   1, MPI_INT, prior_pid, 1, MPI_COMM_WORLD, &recv_req2);
        MPI_CHECK_ERROR(status);

        status = MPI_Wait(&recv_req1,MPI_STATUS_IGNORE);
		MPI_CHECK_ERROR(status);
        status = MPI_Wait(&recv_req2,MPI_STATUS_IGNORE);
		MPI_CHECK_ERROR(status);
		r_st = info->r_start_buffer;
        r_ed = info->r_end_buffer;

		recv_val_num = matB->r_pos[r_ed] - matB->r_pos[r_st];
		
		status = MPI_Isend(matB->gpu_values, send_val_num, MPI_FLOAT, next_pid, 1, MPI_COMM_WORLD, &send_req3);
        MPI_CHECK_ERROR(status);
		status = MPI_Irecv(info->gpu_buffer, recv_val_num, MPI_FLOAT, prior_pid, 1 ,MPI_COMM_WORLD, &recv_req3);
		MPI_CHECK_ERROR(status);
		
        //利用非阻塞MPI_拿到自己需要的新一轮的MatB->values
// #ifdef wwb
// 		data_t *tmp_values = (data_t *)malloc(recv_val_num * sizeof(data_t));
// 		cudaMemcpy(tmp_values, info->gpu_buffer, recv_val_num * sizeof(data_t),cudaMemcpyDeviceToHost);

// 		printf("my p_id is %d\n",getpid());
// 		int signal = 1;
// 		while(signal){
// 			sleep(2);
// 		}
// #endif
        //等待上面的ISend 和 IRecv 都做完
		//printf("before compute\n");
	    compute<<<dimGrids, dimBlock, 0, compute_stream>>>(C_rowIndex, matC->gpu_c_idx, matC->gpu_values, matA->gpu_r_pos, matB->gpu_r_pos, matA->gpu_c_idx, matB->gpu_c_idx, matA->gpu_values, matB->gpu_values, matC->global_nnz, info->r_start, info->r_end, 1.0);
		//传输matB的数据
		
		status = MPI_Wait(&send_req3,MPI_STATUS_IGNORE);
		MPI_CHECK_ERROR(status);
		status = MPI_Wait(&recv_req3,MPI_STATUS_IGNORE);
		MPI_CHECK_ERROR(status);

		cudaStreamSynchronize(compute_stream);
		error = cudaGetLastError();
		CUDACHECK(error);
        //调换buffer指针
        data_t *tmp_values_swap = matB->gpu_values;
        matB->gpu_values = info->gpu_buffer;
        info->gpu_buffer = tmp_values_swap;

        info->r_start = info->r_start_buffer;
        info->r_end   = info->r_end_buffer;

        // MPI_Wait(&recv_req3,MPI_STATUS_IGNORE);
		//MPI_Barrier(MPI_COMM_WORLD);
        
    }
	
    //compute<<<dimGrids, dimBlock>>>(C_rowIndex, matC->gpu_c_idx, matC->gpu_values, matA->gpu_r_pos, matB->gpu_r_pos, matA->gpu_c_idx, matB->gpu_c_idx, matA->gpu_values, matB->gpu_values, matC->global_nnz, info->r_start, info->r_end,1.0);
    cudaDeviceSynchronize();
	error = cudaGetLastError();
	CUDACHECK(error);  
	CUDACHECK(cudaMemcpy(matC->c_idx, matC->gpu_c_idx, (matC->global_nnz) * sizeof(index_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(matC->values, matC->gpu_values, (matC->global_nnz) * sizeof(data_t), cudaMemcpyDeviceToHost));

#ifdef wwb
	char filename[50];
	sprintf(filename, "valuesC%d.txt",p_id);
	FILE * file = fopen(filename,"w");
	for(int i = 0; i < matC->global_nnz; i++){
		fprintf(file, "%lf\n", matC->values[i]);
	}
	fclose(file);
#endif


}

// void spgemm(dist_matrix_t *mat, dist_matrix_t *matB, dist_matrix_t *matC) {


//     //please put your result back in matC->r_pos/c_idx/values in CSR format
//     matC->global_m = mat->global_m;
//     matC->r_pos = (int*)malloc(sizeof(int)*(matC->global_m+1));
    
// }
