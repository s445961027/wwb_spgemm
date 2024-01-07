#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED 1

#include<stdint.h>
#include<stdbool.h>
#include <stdio.h>

#define UNREACHABLE (-1)
typedef int32_t index_t;
typedef float data_t;
#define MPI_DATA MPI_FLOAT
typedef void (*free_func_t)(void*);

/* 
 * global_m: number of rows in the whole input matrix
 * global_n: number of columns in the whole input matrix
 * global_nnz: number of non-zeros in the whole input matrix
 * local_m: number of rows in the current process
 * offset_i: number of rows in previous processes
 * local_nnz: number of non-zeros in the current process
 */
typedef struct {
    int global_m = 0, global_nnz = 0;             /* do not modify */
    
    index_t* r_pos = NULL;
    index_t* c_idx = NULL;
    data_t* values = NULL;
    free_func_t CPU_free = NULL;

    index_t* gpu_r_pos = NULL;
    index_t* gpu_c_idx = NULL;
    data_t* gpu_values = NULL;
    free_func_t GPU_free = NULL;

    void *additional_info = NULL;         /* any information you want to attach */
} dist_matrix_t;


typedef struct {
    //该进程关于所存储的matB行号的 起始行号r_start和结束行号r_end - 1
    //比如0号进程存储了matB的 第 0 - 1 行，一共两行，r_start为0， r_end为2
    int r_start = 0;
    int r_end = 0;
    int r_start_buffer = 0;
    int r_end_buffer = 0;
    data_t *gpu_buffer = NULL;

}additional_info;


#ifdef __cplusplus
extern "C" {
#endif

void preprocess(dist_matrix_t *matA, dist_matrix_t *matB);
void destroy_additional_info(void *additional_info);
void spgemm(dist_matrix_t *mat, dist_matrix_t *matB, dist_matrix_t *res, int p_id, int process_num);

#ifdef __cplusplus
}
#endif

#endif