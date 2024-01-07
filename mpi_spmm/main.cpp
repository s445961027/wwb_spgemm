#include "mpi.h"
#include <sys/time.h>
#include "common.h"
#include <unistd.h>
#include "utils.h"
#include <vector>
extern const char* version_name;

int parse_args(int* reps, int p_id, int argc, char **argv);
int my_abort(int line, int code);
void main_gpufree(void *p);
//extern void getRowNnz(dist_matrix_t *mat, dist_matrix_t* matB, dist_matrix_t* matC, std::vector<index_t>& c_idx);

#define MY_ABORT(ret) my_abort(__LINE__, ret)
#define ABORT_IF_ERROR(ret) CHECK_ERROR(ret, MY_ABORT(ret))
#define ABORT_IF_NULL(ret) CHECK_NULL(ret, MY_ABORT(NO_MEM))
#define INDENT "    "
#define TIME_DIFF(start, stop) 1.0 * (stop.tv_sec - start.tv_sec) + 1e-6 * (stop.tv_usec - start.tv_usec)

//#define wwb


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int my_pid, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // TODO: This function allocates one GPU per process.
    //       You could change the way of device assignment 
    init_one_gpu_per_process(my_pid);

    int warm_up = 0, reps, i, ret;
    dist_matrix_t matA;
    dist_matrix_t matB;
    dist_matrix_t part_matC;
    dist_matrix_t matC;


    double compute_time = 0.0, pre_time = 0.0;
    struct timeval tick0, tick1;

    ret = parse_args(&reps, 0, argc, argv);
    ABORT_IF_ERROR(ret)
    // TODO: if you want to read the matrix in a distributed way, 
    //       or redistribute the matrix after reading it on a particular process,
    //       you could modify here.
    
    ret = read_matrix_distribute_matA(&matA, argv[2], num_procs, my_pid);
    ABORT_IF_ERROR(ret)
    
    ret = read_matrix_distribute_matB(&matB, argv[2], num_procs, my_pid);
    ABORT_IF_ERROR(ret)
    
    part_matC.additional_info = NULL;
    part_matC.CPU_free = free;
    part_matC.GPU_free = main_gpufree;

    matC.additional_info = NULL;
    matC.CPU_free = free;
    matC.GPU_free = main_gpufree;
    
    // if (my_pid == 0) {
    //     printf("Benchmarking %s on %s.\n", version_name, argv[2]);
    //     printf(INDENT"%d process's matA's shape is %d x %d, %d non-zeros, %d run(s)\n", \
    //             matA.global_m, matA.global_m, matA.global_nnz, reps);
    // }

    {
        printf("Benchmarking %s on %s.\n", version_name, argv[2]);
        printf(INDENT"%d process's matA's shape is %d x %d, %d non-zeros, %d run(s)\n", \
                my_pid, matA.global_m, matA.global_m, matA.global_nnz, reps);
    }
    // This is a very naive implementation that only rank 0 does the work, and the others keep idle
    // Thus, only one GPU is utilized here.
    // You should replace it with your own optimized multi-GPU implementation.
    // Load balance across different devices may be important.

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&tick0, NULL);

    // TODO: You should replace it in your distributed way of preprocess.
    if (my_pid == 0) {// only rank 0 does the work
        preprocess(&matA, &matB);
    } else {
        // idle
    }

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&tick1, NULL);
    pre_time += TIME_DIFF(tick0, tick1);

    
    for (int test = 0; test < reps + warm_up; test++) {

        // TODO: Clear the working space and the result matrix for multiple tests
        //       You should modify it in your distributed way.
        if (my_pid == 0) {// only rank 0 has the data
            //destroy_dist_matrix(&matC);
        } else {
            // idle
        }

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tick0, NULL);

        // TODO: You should modify it in your distributed way.
        
        //
        spgemm(&matA, &matB, &part_matC, my_pid, num_procs);
        
        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tick1, NULL);
        if (test >= warm_up) compute_time += TIME_DIFF(tick0, tick1);
    }
    
    // TODO: Here collect the distributed result to rank 0 for correctness check.
    if (my_pid == 0) {
        // recv the local results from other ranks if needed
        //先收集不同进程有matC的多少行
        int process_num = num_procs;
        //dist_matrix_t matC;
        int part_m[process_num];
        //part_r_len是指不同进程所拥有的r_pos的长度，是part_global_m + 1
        int part_r_len[process_num];
        //values_num是指不同进程中所拥有的values的数量
        int part_nnz[process_num];

        int global_m = 0;
        int global_nnz = 0;
        data_t *values;
        index_t *r_pos;
        index_t *r_pos_tmp;
        index_t *c_idx;
        
        //gather_global_m

        
        MPI_Gather(&(part_matC.global_m), 1, MPI_INT, part_m, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&part_matC.global_nnz, 1, MPI_INT, part_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);


        //total_global_m
        for(int i = 0; i < process_num; i++){
            global_m += part_m[i];
            part_r_len[i] = part_m[i] + 1;//每个进程实际拥有的r_pos的长度是 (行数 + 1)
        }

        //用于gather所有进程临时的r_pos的,因为每个进程负责维护的r_pos的长度为part_m + 1
        //所以总长度为global_m + process_num(每个进程都多1，故加process_num)
        r_pos_tmp = (index_t *)malloc((global_m + process_num) * sizeof(index_t));
        
        //最终的r_pos
        r_pos = (index_t *)malloc((global_m + 1) * sizeof(index_t));

        //用于gather所有进程的r_pos时的 起始偏移数组。
        int r_pos_displs[process_num];
        r_pos_displs[0] = 0;

        for(int i = 1; i < process_num; i++){
            r_pos_displs[i] = r_pos_displs[i - 1] + part_r_len[i - 1];
        }

        //利用part_r_len来收集不同进程的part_matC->r_pos
        MPI_Gatherv(part_matC.r_pos,part_matC.global_m+1, MPI_INT, r_pos_tmp, part_r_len, r_pos_displs, MPI_INT, 0, MPI_COMM_WORLD);
        //因为gather得到的所有r_pos长度都是 part_global_m + 1的，并且都是从0开始，需要merge一下
        //如 [0,2,5] [0,3,6]  需要合并成[0,2,5,8,11],才是真正的r_pos
        int start_offset = 0, base = 0, count = 0;
        for(int i = 0; i < process_num; i++){
            for(int j = 0; j < part_m[i]; j++){

                r_pos[count] = r_pos_tmp[start_offset + j] + base;
                count++;
            }

            base += r_pos_tmp[start_offset + part_m[i]];
            start_offset = start_offset + part_m[i] + 1;
        }
        r_pos[global_m] = base;

        //用于收集每个进程的values的数目

        for(int i = 0; i < process_num; i++){
            global_nnz += part_nnz[i];
        }

        values = (data_t *) malloc(global_nnz * sizeof(data_t));
        c_idx  = (index_t *)malloc(global_nnz * sizeof(index_t));
        
        //不同进程在根进程中values/c_idx的起始位置
        int part_values_displs[process_num];
        part_values_displs[0] = 0;
        for(int i = 1; i < process_num; i++){
            part_values_displs[i] = part_values_displs[i - 1] + part_nnz[i - 1];
        }
        //收集C的c_idx和values
        MPI_Gatherv(part_matC.c_idx,  part_matC.global_nnz, MPI_INT,   c_idx,  part_nnz, part_values_displs, MPI_INT,   0, MPI_COMM_WORLD);
        MPI_Gatherv(part_matC.values, part_matC.global_nnz, MPI_FLOAT, values, part_nnz, part_values_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        matC.global_m = global_m;
        matC.global_nnz = global_nnz;
        matC.r_pos = r_pos;
        matC.c_idx = c_idx;
        matC.values = values;
        
#ifdef wwb
        printf("%d before first gatherv for r_pos\n",getpid());
        int signal = 1;
	    while(signal){
		    sleep(2);
	    }    
#endif
        // FILE *file = fopen("rposC.txt","w");
        // for(int i = 0; i < matC.global_m+1; i++){
        //     fprintf(file, "%lf\n", matC.r_pos[i]);
        // }
        // fclose(file);
        // file = fopen("cidxC.txt","w");
        // for(int i = 0; i < matC.global_nnz; i++){
        //     fprintf(file, "%lf\n", matC.c_idx[i]);
        // }
        // fclose(file);
        // file = fopen("valuesC.txt","w");
        // for(int i = 0; i < matC.global_nnz; i++){
        //     fprintf(file, "%lf\n", matC.values[i]);
        // }
        // fclose(file);
        // Rank 0 check correctness
        printf(INDENT"Checking.\n");
        ret = check_answer(&matC, argv[3]);
        if(ret == 0) {
            printf("\e[1;32m"INDENT"Result validated.\e[0m\n");
        } else {
            fprintf(stderr, "\e[1;31m"INDENT"Result NOT validated.\e[0m\n");
            MY_ABORT(ret);
        }
        printf(INDENT INDENT"preprocess time = %lf s\n", pre_time);
        printf("\e[1;34m"INDENT INDENT"compute time = %lf s\e[0m\n", compute_time / reps);
    } else {
        // send the local result to rank 0 if needed

        MPI_Gather(&part_matC.global_m, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        //用于收集每个进程的values的数目
        MPI_Gather(&part_matC.global_nnz, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(part_matC.r_pos, part_matC.global_m+1, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(part_matC.c_idx,  part_matC.global_nnz, MPI_INT,   NULL, NULL, NULL, MPI_INT,   0, MPI_COMM_WORLD);
        MPI_Gatherv(part_matC.values, part_matC.global_nnz, MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
#ifdef wwb
        printf("%d before first gatherv for r_pos\n",getpid());
        int signal = 1;
	    while(signal){
		    sleep(2);
	    }    

#endif
    
    
    }

    destroy_dist_matrix(&matA);
    destroy_dist_matrix(&matB);
    destroy_dist_matrix(&matC);
    destroy_dist_matrix(&matC);
    
    MPI_Finalize();
    return 0;
}

void main_gpufree(void *p) {
    cudaFree(p);
}

void print_help(const char *argv0, int p_id) {
    if(p_id == 0) {
        printf("\e[1;31mUSAGE: %s <repetitions> <input-file>\e[0m\n", argv0);
    }
}

int parse_args(int* reps, int p_id, int argc, char **argv) {
    int r;
    if(argc < 3) {
        print_help(argv[0], p_id);
        return 1;
    }
    r = atoi(argv[1]);
    if(r <= 0) {
        print_help(argv[0], p_id);
        return 1;
    }
    *reps = r;
    return SUCCESS;
}

int my_abort(int line, int code) {
    fprintf(stderr, "\e[1;33merror at line %d, error code = %d\e[0m\n", line, code);
    return fatal_error(code);
}

int fatal_error(int code) {
    exit(code);
    return code;
}
