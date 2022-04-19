#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cassert>
#define SHMEM_SIZE 32
using namespace std;

/********************************************************************************************/
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
void info()
{
    int iDev = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, iDev);
    printf("Device %d: %s\n", iDev, iProp.name);
    printf("Number of multiprocessors: %d\n", iProp.multiProcessorCount);
    printf("Total amount of constant memory: %4.2f KB\n", iProp.totalConstMem / 1024.0);
    printf("Total amount of shared memory per block: %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
    printf("Total number of registers available per block: %d\n", iProp.regsPerBlock);
    printf("Maximum number of threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor : % d\n ", iProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor / 32);
}
/**************************************************************************************************/
// initialize an Array with random floating point numbers
void initArray(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}
/**************************************************************************************************/
void print(float *A, const int size)
{
    cout << "\n"
         << endl;
    for (int i = 0; i < size; i++)
        cout << A[i] << ", ";
    cout << endl;
}
/**************************************************************************************************/
//a recursive implementation of the interleaved pair approach of reduction for CPU
float recursiveReduce(float *data, int const size)
{
    // terminate check
    if (size == 1)
        return data[0];
    // renew the stride
    int const stride = size / 2;
    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }
    // call recursively
    return recursiveReduce(data, stride);
}
/**************************************************************************************************/
__global__ void sum_reduction_global(float *g_idata, float *g_odata, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n)
    {
        atomicAdd(g_odata, g_idata[tid]);
        tid += blockDim.x * gridDim.x;
    }
}
/**************************************************************************************************/
__global__ void sum_reduction_shared(float *g_idata, float *g_odata, size_t n)
{
    // Allocate shared memory
    __shared__ float partial_sum[SHMEM_SIZE];
    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // Store first partial result instead of just the elements
    partial_sum[threadIdx.x] = g_idata[i] + g_idata[i + blockDim.x];
    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s)
        {
            atomicAdd(&partial_sum[threadIdx.x], partial_sum[threadIdx.x + s]);
        }
    }
    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = partial_sum[0];
    }
}
/**************************************************************************************************/

int main()
{
    info();
    //Array size
    const int N = 1 << 16;
    float *Arr;
    Arr = (float *)malloc(N * sizeof(float)); // assigning memory to Array
    initArray(Arr, N);                        // initializing random
    //print(Arr, N);
    // Timing The CPU version of recursive
    double s1 = omp_get_wtime();
    float sum_cpu = recursiveReduce(Arr, N);
    double e1 = omp_get_wtime();
    cout << "\n The sum of all the Array elemnts = " << sum_cpu << "\n And it took " << 1.e3 * (e1 - s1) << "ms" << endl;

    float *d_arr, *d_sum, *h_sum;           // pointers for GPU version of recursive using atomic in global
    h_sum = (float *)malloc(sizeof(float)); // host pointer to the result sum

    CUDADEBUG(cudaMalloc((void **)&d_arr, N * sizeof(float)));
    CUDADEBUG(cudaMalloc((void **)&d_sum, sizeof(float)));
    CUDADEBUG(cudaMemcpy(d_arr, Arr, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(8, 1);
    dim3 grid((N - block.x - 1) / block.x, 1);

    // timing the recurcive kernel with Atomic in global
    double s2 = omp_get_wtime();
    sum_reduction_global<<<grid, block>>>(d_arr, d_sum, N * sizeof(float));
    CUDADEBUG(cudaDeviceSynchronize());
    double e2 = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    cout << "\n The GPU recurcive sum in global = " << *h_sum << " and in took " << 1.e3 * (e2 - s2) << " ms" << endl;

    float *d_sum_s, *h_sum_s;                     // pointers to the GPU version of recurcive using Atomic in shared memory
    h_sum_s = (float *)malloc(N * sizeof(float)); //host pointer to the result
    CUDADEBUG(cudaMalloc((void **)&d_sum_s, N * sizeof(float)));

    //timing the recurcive kerenl with Atomic in shared memory;
    double s3 = omp_get_wtime();
    sum_reduction_shared<<<1, 1>>>(d_arr, d_sum_s, N * sizeof(float));
    CUDADEBUG(cudaDeviceSynchronize());
    double e3 = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_sum_s, d_sum_s, N * sizeof(float), cudaMemcpyDeviceToHost));
    cout << "\n The GPU sum_shared = " << h_sum_s[0] << " and it took " << 1.e3 * (e3 - s3) << " ms" << endl;
}