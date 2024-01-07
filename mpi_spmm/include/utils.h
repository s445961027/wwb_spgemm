#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED 1

#include "common.h"
#include "cuda.h"
#include "cuda_runtime.h"

#define CHECK(err, err_code) if(err) { return err_code; }
#define CHECK_ERROR(ret, err_code) CHECK(ret != 0, err_code)
#define CHECK_NULL(ret, err_code) CHECK(ret == NULL, err_code)
#define CUDACHECK(call)                             \
do                                              \
{                                               \
    const cudaError_t error_code = call;        \
    if(error_code != cudaSuccess)               \
    {                                           \
        printf("CUDA Error:\n");                \
        printf("File        %s\n",__FILE__);    \
        printf("Line        %d\n",__LINE__);    \
        printf("Error code: %d\n",error_code);  \
        printf("Error text: %s\n",cudaGetErrorString(error_code));\
        exit(1);                            \
    }                                       \
}while(0)

#ifdef __cplusplus
extern "C" {
#endif

int fatal_error(int code);

int read_matrix_default(dist_matrix_t *mat, const char* filename);
void destroy_dist_matrix(dist_matrix_t *mat);

int read_vector(dist_matrix_t *mat, const char* filename, const char* suffix, int n, data_t* x);
int read_matrix_distribute_matA(dist_matrix_t *mat, const char *filename, int process_num, int pid);
int read_matrix_distribute_matB(dist_matrix_t *mat, const char *filename, int process_num, int pid);
int check_answer(dist_matrix_t *mat, const char* filename);
void init_one_gpu_per_process(int my_pid);

#ifdef __cplusplus
}
#endif

#define SUCCESS cudaSuccess
#define NO_MEM cudaErrorMemoryAllocation
#define IO_ERR 3

#endif