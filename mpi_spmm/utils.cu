#include <float.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <unistd.h>
#include <cusparse.h>
#include "./include/common.h"
#include "utils.h"


#define EPS_TOL 1e-4
//#define wwb
#define CHECK_AND_SET(cond, state) if(cond) {state;}
#define CHECK_AND_BREAK(cond, state) if(cond) {state;break;}

typedef FILE *file_t;

int clean(int ret, void *p) {
    free(p);
    return ret;
}
                
int clean_file(int ret, file_t file) {
    fclose(file);
    return ret;
}

void gpu_free(void *p) {
    cudaFree(p);
}

//假如有两个进程，每个进程读一半A
int read_matrix_distribute_matA(dist_matrix_t *mat, const char *filename, int process_num, int pid){
    
    file_t file;
    int global_m, global_nnz;
    int ret, count;
    index_t *r_pos;
    index_t *c_idx;
    data_t *values;
    index_t *gpu_r_pos;
    index_t *gpu_c_idx;
    data_t *gpu_values;
    
    file = fopen(filename, "rb");
    CHECK(file == NULL, IO_ERR)

    count = fread(&global_m, sizeof(index_t), 1, file);
    CHECK(count != 1, IO_ERR)
    
    //global_nnz = r_pos[global_m]
    ret = fseek(file, sizeof(index_t) * (global_m + 1), SEEK_SET);
    count = fread(&global_nnz,sizeof(index_t), 1, file);
    //printf("global nnz is %d\n",global_nnz);
    //根据进程总数量，每个进程读取一部分的matA
    int part_m, before_m;
    if( pid != process_num - 1){
        part_m = (global_m - 1) / process_num + 1;
    }
    else{
        part_m = global_m - pid * ((global_m - 1) / process_num + 1);
    }

    before_m = pid * ((global_m - 1) / process_num + 1);
    
    r_pos = (index_t *)malloc(sizeof(index_t) * (part_m + 1));
    CHECK(r_pos == NULL, NO_MEM)
    //调整文件指针位置，指向当前r_pos的起始地址
    ret = fseek(file, sizeof(index_t) * (1 + before_m), SEEK_SET);//注意，这里需要把global_m也考虑上,故加了1
    //注意要多读一个，因为r_pos确实是比实际行数多1
    count = fread(r_pos, sizeof(index_t), part_m + 1, file);
    
    
    
    CHECK(count != part_m + 1, IO_ERR)
    
    c_idx =  (index_t *)malloc( (r_pos[part_m] - r_pos[0]) * sizeof (index_t));
    values = (data_t  *)malloc( (r_pos[part_m] - r_pos[0]) * sizeof (data_t) );
    //c_idx的初始位置 (global_m + 1)是mat->r_pos的大小，再+1是因为global_m占4B
    //然后再加当前c_id的初始偏移，当前部分的c_id是由r_pos[end] - r_pos[0]得到的
    //printf("%d debug here,size is %d\n",pid,r_pos[part_m] - r_pos[0]);

#ifdef wwb
    printf("my pid is: %d\n",getpid());
    int signal = 1;
    while(signal){
        sleep(2);
    }

#endif
    CHECK(c_idx == NULL, NO_MEM)
    
    CHECK(values == NULL,NO_MEM)
    
    int c_idx_offset_inFile = (global_m + 1) + 1 + r_pos[0];
    //调整文件指针位置，指向当前的c_idx起始地址
    ret = fseek(file, c_idx_offset_inFile * sizeof(index_t), SEEK_SET);
    count = fread(c_idx, sizeof(index_t), r_pos[part_m] - r_pos[0], file);

    
    CHECK(count != r_pos[part_m] - r_pos[0], IO_ERR)
    //需要读入的values的初始位置，(global_m + 1)是mat->r_pos的大小，再+1是因为global_m占4B
    //global_nnz是c_id的总量，然后再加上当前value的初始偏移，和cid的计算方式一样，因为和cid是对应的
    int value_offset_inFile = (global_m + 1) + 1 + global_nnz + r_pos[0];
    ret = fseek(file, value_offset_inFile * sizeof(index_t), SEEK_SET);
    count = fread(values, sizeof(data_t), r_pos[part_m] - r_pos[0],file);
    CHECK(count != r_pos[part_m] - r_pos[0], IO_ERR)
    
    fclose(file);
    
    //因读取的是部分r_pos,需将r_pos调整为r_pos[0] = 0;
    int offset = r_pos[0];
    for(int i = 0; i < part_m + 1; i++){
        r_pos[i] -= offset;
    }
#ifdef wwb
    printf("process %d,r_pos[0] is %d,r_pos[part_m] is %d\n",pid,r_pos[0],r_pos[part_m]);
#endif
    cudaMalloc(&gpu_r_pos, (part_m + 1) * sizeof(index_t));
    cudaMalloc(&gpu_c_idx, (r_pos[part_m] - r_pos[0]) * sizeof (index_t));
    cudaMalloc(&gpu_values,(r_pos[part_m] - r_pos[0]) * sizeof (data_t));
    
    cudaMemcpy(gpu_r_pos, r_pos, (part_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c_idx,  c_idx,  (r_pos[part_m] - r_pos[0]) * sizeof (index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_values, values, (r_pos[part_m] - r_pos[0]) * sizeof (data_t),  cudaMemcpyHostToDevice);

    mat->global_m = part_m;
    mat->global_nnz = r_pos[part_m] - r_pos[0];
    mat->r_pos = r_pos;//这里考虑成r_pos[0] = 0?
    mat->c_idx = c_idx;
    mat->values = values;
    mat->gpu_r_pos = gpu_r_pos;
    mat->gpu_c_idx = gpu_c_idx;
    mat->gpu_values = gpu_values;



    return SUCCESS;
}

int read_matrix_distribute_matB(dist_matrix_t *mat, const char *filename, int process_num, int pid){

    file_t file;
    int global_m, global_nnz;
    int ret, count;
    index_t *r_pos;
    index_t *c_idx;
    data_t *values;
    index_t *gpu_r_pos;
    index_t *gpu_c_idx;
    data_t *gpu_values;
    file = fopen(filename, "rb");

    fseek(file,0,SEEK_SET);
    
    CHECK(file == NULL, IO_ERR);
    count = fread(&global_m, sizeof(index_t), 1, file);
    CHECK(count != 1, IO_ERR)

    //每个进程都维护全部的matB->r_pos，matB->c_idx


    r_pos = (index_t*)malloc(sizeof(index_t) * (global_m + 1));
    CHECK(r_pos == NULL, NO_MEM)

    count = fread(r_pos, sizeof(index_t), global_m + 1, file);
    CHECK(count != global_m + 1, IO_ERR)
    global_nnz = r_pos[global_m];

    c_idx = (index_t*)malloc(sizeof(index_t) * global_nnz);
    CHECK(c_idx == NULL, NO_MEM)
    count = fread(c_idx, sizeof(index_t), global_nnz, file);

    //每个进程负责存B的某几行的value
    int part_m;
    if( pid != process_num - 1){
        part_m = global_m / process_num;
    }
    else{
        part_m = global_m -  pid * (global_m / process_num);
    }

    //不同进程存B部分行,行数为part_m
    index_t r_st = pid * (global_m / process_num);
    index_t r_ed = r_st + part_m;

    int val_num = r_pos[r_ed] - r_pos[r_st];

    values = (data_t  *)malloc(val_num * sizeof(data_t));
    
    int value_offset_inFile = (global_m + 1) + 1 + global_nnz + r_pos[r_st];
    
    ret = fseek(file,value_offset_inFile * sizeof(index_t), SEEK_SET);
    fread(values, sizeof(data_t), val_num, file);
    fclose(file);

    //这里先假设一下buffer的大小，后续再调,
    //buffer1是gpu_values,buffer2是用于双缓冲
    data_t *gpu_buffer;
    CUDACHECK(cudaMalloc(&gpu_r_pos,   sizeof(index_t) * (global_m + 1)));
    CUDACHECK(cudaMalloc(&gpu_c_idx,   sizeof(index_t) * (global_nnz)));
    CUDACHECK(cudaMalloc(&gpu_values, global_nnz * sizeof(data_t)));
    CUDACHECK(cudaMalloc(&gpu_buffer, global_nnz * sizeof(data_t)));

    cudaMemcpy(gpu_values, values, val_num * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_r_pos,   r_pos,  (global_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c_idx,   c_idx,  global_nnz * sizeof(index_t), cudaMemcpyHostToDevice);

    additional_info *extra_info = (additional_info *)malloc(sizeof(additional_info));
    extra_info->r_start    = r_st;
    extra_info->r_end      = r_ed;
    extra_info->gpu_buffer = gpu_buffer;
    
    mat->global_m = global_m;
    mat->r_pos  = r_pos;
    mat->c_idx  = c_idx;
    mat->values = values;
    mat->gpu_r_pos  = gpu_r_pos;
    mat->gpu_c_idx  = gpu_c_idx;
    mat->gpu_values = gpu_values;//部分B的values
    mat->global_nnz = global_nnz;
    mat->additional_info = extra_info;

    return SUCCESS;
}


int read_matrix_default(dist_matrix_t *mat, const char* filename) {
    file_t file;
    int global_m, global_nnz;
    int ret, count;
    index_t *r_pos;
    index_t *c_idx;
    data_t *values;
    index_t *gpu_r_pos;
    index_t *gpu_c_idx;
    data_t *gpu_values;

    file = fopen(filename, "rb");
    CHECK(file == NULL, IO_ERR)

    count = fread(&global_m, sizeof(index_t), 1, file);
    CHECK(count != 1, IO_ERR)

    r_pos = (index_t*)malloc(sizeof(index_t) * (global_m + 1));
    CHECK(r_pos == NULL, NO_MEM)

    count = fread(r_pos, sizeof(index_t), global_m + 1, file);
    CHECK(count != global_m + 1, IO_ERR)
    global_nnz = r_pos[global_m];
    printf("global nnz is %d\n",global_nnz);
    c_idx = (index_t*)malloc(sizeof(index_t) * global_nnz);
    CHECK(c_idx == NULL, NO_MEM)
    values = (data_t*)malloc(sizeof(data_t) * global_nnz);
    CHECK(values == NULL, NO_MEM)

    count = fread(c_idx, sizeof(index_t), global_nnz, file);
    CHECK(count != global_nnz, IO_ERR)
    count = fread(values, sizeof(data_t), global_nnz, file);
    CHECK(count != global_nnz, IO_ERR)

    fclose(file);

    ret = cudaMalloc(&gpu_r_pos, sizeof(index_t) * (global_m + 1));
    CHECK_ERROR(ret, ret)
    ret = cudaMalloc(&gpu_c_idx, sizeof(index_t) * global_nnz);
    CHECK_ERROR(ret, ret)
    ret = cudaMalloc(&gpu_values, sizeof(data_t) * global_nnz);
    CHECK_ERROR(ret, ret)
    
    cudaMemcpy(gpu_r_pos, r_pos, sizeof(index_t) * (global_m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c_idx, c_idx, sizeof(index_t) * global_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_values, values, sizeof(data_t)  * global_nnz, cudaMemcpyHostToDevice);

    mat->global_m = global_m;
    mat->global_nnz = global_nnz;
    mat->r_pos = r_pos;
    mat->c_idx = c_idx;
    mat->values = values;
    mat->gpu_r_pos = gpu_r_pos;
    mat->gpu_c_idx = gpu_c_idx;
    mat->gpu_values = gpu_values;
    mat->additional_info = NULL;
    mat->CPU_free = free;
    mat->GPU_free = gpu_free;
    return SUCCESS;
}

char *cancat_name(const char *a, const char *b) {
    int l1 = strlen(a) - 4, l2 = strlen(b);
    char *c = (char*)malloc(sizeof(char) * (l1 + l2 + 1));
    if(c != NULL) {
        memcpy(c, a, l1 * sizeof(char));
        memcpy(c + l1, b, l2 * sizeof(char));
        c[l1 + l2] = '\0';
    }
    return c;
}


int read_vector(dist_matrix_t *mat, const char* filename, const char* suffix, int n, data_t* x) {
    char *new_name;
    file_t file;
    int count, i, m = mat->global_m;
    new_name = cancat_name(filename, suffix);
    CHECK(new_name == NULL, NO_MEM)

    file = fopen(new_name, "rb");
    CHECK(new_name == NULL, clean(IO_ERR, new_name))

    for(i = 0; i < n; ++i) {
        count = fread(x + m * i, sizeof(data_t), m, file);
        CHECK(count != m, clean(clean_file(IO_ERR, file), new_name))
    }
    return clean(clean_file(SUCCESS, file), new_name);
}

int check_answer(dist_matrix_t *mat, const char* filename) {
    FILE *file;
    int res_m, res_nnz;
    int *res_r_pos, *res_c_idx;
    float *res_values;
    int i,j;
    int x;

    file = fopen(filename, "rb");
    fread(&res_m, sizeof(index_t), 1, file);
    res_r_pos = (index_t*)malloc(sizeof(index_t) * (res_m + 1));
    fread(res_r_pos, sizeof(index_t), res_m + 1, file);
    res_nnz = res_r_pos[res_m];
    res_c_idx = (index_t*)malloc(sizeof(index_t) * res_nnz);
    res_values = (data_t*)malloc(sizeof(data_t) * res_nnz);
    fread(res_c_idx, sizeof(index_t), res_nnz, file);
    fread(res_values, sizeof(data_t), res_nnz, file);
    fclose(file);

    int *resid, *pos;
    resid = (int*) malloc(sizeof(int)*(res_m+1));
    pos = (int*) malloc(sizeof(int)*(res_m+1));
    int resi_num;
    float *resi_val;
    float *std_resi;
    float *res_row, *res_col;
    float std;
    res_row = (float*)malloc(sizeof(float)*(res_m+1));
    res_col = (float*)malloc(sizeof(float)*(res_m+1));
    resi_val = (float*)malloc(sizeof(float)*(res_m+1));
    std_resi = (float*)malloc(sizeof(float)*(res_m+1));

    for (i = 0; i < res_m; i++){
        res_row[i] = 0;
        res_col[i] = 0;
        pos[i] = -1;
    }
    for (i = 0; i < res_m; i++) {
        for (j = mat->r_pos[i]; j < mat->r_pos[i+1]; j++) {
            x = mat->c_idx[j];
            res_row[i] = res_row[i] + (mat->values[j]) * (mat->values[j]);
            res_col[x] = res_col[x] + (mat->values[j]) * (mat->values[j]);
        }
    }
    for (i = 0; i < res_m; i++) {
        res_row[i] = sqrt(res_row[i]);
        res_col[i] = sqrt(res_col[i]);
    }

    for (i = 0; i < res_m; i++) {
        resi_num = 0;
        for (j = mat->r_pos[i]; j < mat->r_pos[i+1]; j++) {
            x = mat->c_idx[j];
            if (pos[x] != i) {
                pos[x] = i;
                resid[resi_num] = x;
                resi_num++;
                std_resi[x] = 0;
                resi_val[x] = 0; 
            }
            resi_val[x] = resi_val[x] + mat->values[j];
            std_resi[x] = mat->values[j];
        }
        for (j = res_r_pos[i]; j < res_r_pos[i+1]; j++) {
            x = res_c_idx[j];
            if (pos[x] != i) {
                pos[x] = i;
                resid[resi_num] = x;
                resi_num++;
                resi_val[x] = 0; 
                std_resi[x] = 0;
            }
            resi_val[x] = resi_val[x] - res_values[j];
            if (std_resi[x] == 0) std_resi[x] = res_values[j];
        }
        for (j = 0; j < resi_num; j++) {
            x = resid[j];
            if (resi_val[x] < 0) resi_val[x] = -resi_val[x];
            if (std_resi[x] < 0) std_resi[x] = -std_resi[x];
            std = res_row[i] * res_col[x] * 1e-3 + 1e-6;
            if (resi_val[x] > std) return -1;
        }
    }
    return 0;
}

void destroy_dist_matrix(dist_matrix_t *mat) {
    if(mat->additional_info != NULL){
        destroy_additional_info(mat->additional_info);
        mat->additional_info = NULL;
    }
    if(mat->CPU_free != NULL) {
        if(mat->r_pos != NULL){
            mat->CPU_free(mat->r_pos);
            mat->r_pos = NULL;
        }
        if(mat->c_idx != NULL){
            mat->CPU_free(mat->c_idx);
            mat->c_idx = NULL;
        }
        if(mat->values != NULL){
            mat->CPU_free(mat->values);
            mat->values = NULL;
        }
    }
    if(mat->GPU_free != NULL) {
        if(mat->gpu_r_pos != NULL){
            mat->GPU_free(mat->gpu_r_pos);
            mat->gpu_r_pos = NULL;
        }
        if(mat->gpu_c_idx != NULL){
            mat->GPU_free(mat->gpu_c_idx);
            mat->gpu_c_idx = NULL;
        }
        if(mat->gpu_values != NULL){
            mat->GPU_free(mat->gpu_values);
            mat->gpu_values = NULL;
        }
    }
}

void init_one_gpu_per_process(int my_pid)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int dev = my_pid & 0b1;// 注意每节点上有两块GPU

	// for (int dev = 0; dev < deviceCount; dev++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		// if (dev == 0)
		// 	if (deviceProp.minor = 9999 && deviceProp.major == 9999)
		// 		printf("\n");
		// printf("Device%d:\"%s\"\n", dev, deviceProp.name);
		// int driver_version = 0, runtime_version = 0;
		// cudaDriverGetVersion(&driver_version);
		// printf("CUDA Driver version:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		// cudaRuntimeGetVersion(&runtime_version);
		// printf("CUDA Runtim version:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		// printf("Device Compute capability:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
		// printf("Total amount of Global Memory:                  %zu bytes\n", deviceProp.totalGlobalMem);
		// printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
		//printf("Total amount of Constant Memory:                %zu bytes\n", deviceProp.totalConstMem);
		// printf("Total amount of Shared Memory per block:        %zu bytes\n", deviceProp.sharedMemPerBlock);
		// printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
		// printf("Warp size:                                      %d\n", deviceProp.warpSize);
		// printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
		// printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
		// printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		// printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		// printf("Maximum memory pitch:                           %zu bytes\n", deviceProp.memPitch);
		//printf("Texture alignmemt:                              %zu bytes\n", deviceProp.texturePitchAlignment);
		//printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
		//printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
		// printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
		// printf("Concurrent Kernel Execution:                    %d\n", deviceProp.concurrentKernels);
		// printf("can Map Host Memory:                            %d\n", deviceProp.canMapHostMemory);
		// printf("can Use Host Pointer For Registered Mem:        %d\n", deviceProp.canUseHostPointerForRegisteredMem);
		// printf("\n", deviceProp.)
		// printf("\n");
	// }

	cudaSetDevice(dev);
	printf("Rank %d can access %d devices, dev_id %d name %s\n", my_pid, deviceCount, dev, deviceProp.name);
}

