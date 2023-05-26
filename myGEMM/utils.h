
#include <stdlib.h>
#include <stdio.h>
#include <time.h>




void genRandomMatrix(float* A, int M, int N) {
    srand(time(NULL));   // Initialization, should only be called once.
    float a = 5.0;
    for ( int i = 0; i < M; i ++ ) {
        for (int j = 0; j < N; j ++) {
            A[i * N + j] = (float) rand() / ((float)RAND_MAX / a);
        }
    }
}


void copyMatrix(float* des, float* src, int M, int N) {
    for ( int i = 0; i < M; i ++ ) {
        for (int j = 0; j < N; j ++) {
            des[i * N + j] = src[i * N + j];
        }
    }
}