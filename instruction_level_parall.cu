#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>
using namespace std;
//===================================================================================================//
/*
The following macro is for cheking CUDA Errors while allocationg memoryin Device and transfering the data
*/
#define DEBUG(str) std::cerr << "\033[1;37m" << __FILE__ << ":" << __LINE__ << ": \033[1;31merror:\033[0m " << str << std::endl;

#define CUDADEBUG(call)                     \
    {                                       \
        const cudaError_t err = call;       \
        if (err != cudaSuccess)             \
            DEBUG(cudaGetErrorString(err)); \
    }
/****************************************************************************************************/
//======================================================================================================
/* The following is the kernel which computes one independent  instruction in one itiration */
__global__ void ilp_1(float *h_A, const int N_ITR)
{
    float a = 1.00;
    float b = 2.00;
    float c = 3.00;
    // unrolling the Loop for more improvment .
#pragma unroll 16
    for (int i = 0; i < N_ITR; i++)
        a = a * b + c;
    h_A[threadIdx.x] = a; // putting the result in the global memory
}

//======================================================================================================

/* the following is the kernel which excecuts 4 independent instruction per iteration */
__global__ void ilp_4(float *h_A, float *h_D, float *h_E, float *h_F, const int N_ITR)
{
    float a = 1.00;
    float b = 2.00;
    float c = 3.00;
    float d = 4.00;
    float e = 5.00;
    float f = 6.00;
#pragma unroll 16
    for (int i = 0; i < N_ITR; i++)
    {
        a = a * b + c;
        d = d * b + c;
        e = e * b + c;
        f = f * b + c;
    }
    h_A[threadIdx.x] = a;
    h_D[threadIdx.x] = d;
    h_E[threadIdx.x] = e;
    h_F[threadIdx.x] = f;
}

//============================================================================================================
int main(int argc, char **argv)
{
    int BlockDX = atoi(argv[1]); // taking the number of threads per block through the input
    const int N_iters = 1 << 15; // number of iterations of loops in kernels

    float *d_A, *d_D, *d_E, *d_F; // pointer to device memory arryas
    const size_t Bytes = BlockDX * sizeof(float);
    CUDADEBUG(cudaMalloc(&d_A, Bytes));
    CUDADEBUG(cudaMalloc(&d_D, Bytes));
    CUDADEBUG(cudaMalloc(&d_E, Bytes));
    CUDADEBUG(cudaMalloc(&d_F, Bytes));

    double start1 = omp_get_wtime();
    ilp_1<<<1, BlockDX>>>(d_A, N_iters);
    CUDADEBUG(cudaDeviceSynchronize());
    double end1 = omp_get_wtime();
    cout << "\n The ILP_1 kernel: execution time per one operation : " << 1.e9 * (end1 - start1) / (N_iters * BlockDX * 2) << " ms" << endl;

    double start2 = omp_get_wtime();
    ilp_4<<<1, BlockDX>>>(d_A, d_D, d_E, d_F, N_iters);
    CUDADEBUG(cudaDeviceSynchronize());
    double end2 = omp_get_wtime();
    cout << "\n The ILP_4 kernel: execution time per one operation : " << 1.e9 * (end2 - start2) / (N_iters * BlockDX * 2 * 4) << " ms" << endl;
    cout << endl;

    CUDADEBUG(cudaFree(d_A));
    CUDADEBUG(cudaFree(d_D));
    CUDADEBUG(cudaFree(d_E));
    CUDADEBUG(cudaFree(d_F));

    cudaDeviceReset();
}
