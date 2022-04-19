#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cassert>
#define ILP 1024
#define SHMEM_SIZE 1024
#define BS 1024
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
        //ip[i] = (float)(rand() & 0xFF) / 10.0f;
        ip[i] = 1;
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
/***********************************************************************************************/
__global__ void sum_reduction_global_atomic(float *g_idata, float *g_odata, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n)
    {
        atomicAdd(g_odata, g_idata[tid]);
        tid += blockDim.x * gridDim.x;
    }
}
/***********************************************************************************************/
__global__ void sum_reduction_shared_atomic(float *idata, float *odata, size_t size)
{
    __shared__ float blocksum[1];
    size_t bid = ILP * blockIdx.x * gridDim.x;
    size_t tid = ILP * threadIdx.x;
    // sub-reduction on thread-level with ILP
    float tsum = 0.0;
    for (size_t i = 0; i < ILP; ++i)
    {
        if (tid + bid + i > size)
            break;
        tsum += idata[tid + bid + i];
    }
    __syncthreads();
    // sub-reduction on block-level
    atomicAdd(blocksum, tsum);
    __syncthreads();
    // final reduction on grid-level in global memory
    if (tid == 0)
        atomicAdd(odata, blocksum[0]);
}
/***********************************************************************************************/
__global__ void sum_reduction_divergent(float *idata, float *odata)
{
    __shared__ float partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Load elements into shared memory
    partial_sum[threadIdx.x] = idata[tid];
    __syncthreads();
    // Iterate of log base 2 the block dimension
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (threadIdx.x % (2 * s) == 0)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Let the thread 0 for this block write it's result to main memory
    // Result is indexed by this block
    if (threadIdx.x == 0)
    {
        odata[blockIdx.x] = partial_sum[0];
    }
}

/************************************************************************************************/
__global__ void sum_reduction_BCon(float *idata, float *odata)
{
    __shared__ float partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Load elements into shared memory
    partial_sum[threadIdx.x] = idata[tid];
    __syncthreads();
    // Increase the stride of the access until we exceed the CTA dimensions
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        // Change the indexing to be sequential threads
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x)
        {
            partial_sum[index] += partial_sum[index + s];
        }
        __syncthreads();
    }
    // Let the thread 0 for this block write it's result to main memory
    // Result is indexed by this block
    if (threadIdx.x == 0)
    {
        odata[blockIdx.x] = partial_sum[0];
    }
}
/**********************************************************************************************/

__global__ void sum_reduction_NoBCon(float *idata, float *odata)
{
    __shared__ float partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Load elements into shared memory
    partial_sum[threadIdx.x] = idata[tid];
    __syncthreads();
    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0)
    {
        odata[blockIdx.x] = partial_sum[0];
    }
}
/***********************************************************************************************/
__global__ void sum_reduction_Noidle(float *idata, float *odata)
{
    __shared__ float partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // Store first partial result instead of just the elements
    partial_sum[threadIdx.x] = idata[i] + idata[i + blockDim.x];
    __syncthreads();
    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Let the thread 0 for this block write it's result to main memory
    // Result is indexed by this blockid
    if (threadIdx.x == 0)
    {
        odata[blockIdx.x] = partial_sum[0];
    }
}
/********************************************************************************************/
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, int tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}
/****************************************************************/
__global__ void sum_reduction_warp(float *v, float *v_r)
{
    __shared__ float partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // Store first partial result instead of just the elements
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();
    // Start at 1/2 block stride and divide by two each iteration
    // Stop early (call device function instead)
    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32)
    {
        warpReduce<BS>(partial_sum, threadIdx.x);
    }
    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0)
    {
        v_r[blockIdx.x] = partial_sum[0];
    }
}
/*****************************************************************************************/
__device__ float warpreduceshuffle(float sum)
{
    sum += __shfl_xor(sum, 16);
    sum += __shfl_xor(sum, 8);
    sum += __shfl_xor(sum, 4);
    sum += __shfl_xor(sum, 2);
    sum += __shfl_xor(sum, 1);
    return sum;
}
/**************************************************************/
__global__ void sum_reduction_shfl(float *idata, float *odata, unsigned int n)
{
    __shared__ float smem[SHMEM_SIZE];
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx > n)
        return;
    int sum = idata[idx];
    int laneid = threadIdx.x % 64;
    int warpid = threadIdx.x / 64;
    sum = warpreduceshuffle(sum);
    if (laneid == 0)
        smem[warpid] = sum;
    __syncthreads();
    sum = (threadIdx.x < SHMEM_SIZE) ? smem[laneid] : 0;
    if (warpid == 0)
        sum = warpreduceshuffle(sum);
    if (threadIdx.x == 0)
        odata[blockIdx.x] = sum;
}

/*#############################################################################################*/

int main(int argc, char **argv)
{
    info();
    const int N = 1 << 22;
    size_t bytes = N * sizeof(float);

    float *h_arr;
    h_arr = (float *)malloc(bytes);
    initArray(h_arr, N);

    double t_ = omp_get_wtime();
    float sum_cpu = recursiveReduce(h_arr, N);
    double te = omp_get_wtime();

    cout << "\n The CPU version of reduction took :: " << 1.e3 * (te - t_) << " ms"
         << "\t \t "
         << "sum = " << sum_cpu << endl;

    /*.......................................................*/
    /*********************************************************/
    dim3 block(BS);
    dim3 grid((N / sizeof(float)) / BS + 1);
    dim3 grididle(((N / sizeof(float)) / BS) / 2 + 1);
    /*.......................................................*/
    /*********************************************************Timing the global atomic kernel**/

    float *d_ga_in, *d_ga_out, *h_ga_out;
    CUDADEBUG(cudaMalloc((void **)&d_ga_in, bytes));
    CUDADEBUG(cudaMalloc((void **)&d_ga_out, sizeof(float)));
    h_ga_out = (float *)malloc(sizeof(float));
    CUDADEBUG(cudaMemcpy(d_ga_in, h_arr, bytes, cudaMemcpyHostToDevice));
    double t1 = omp_get_wtime();
    sum_reduction_global_atomic<<<grid, block>>>(d_ga_in, d_ga_out, N / sizeof(float));
    CUDADEBUG(cudaDeviceSynchronize());
    double t1e = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_ga_out, d_ga_out, sizeof(float), cudaMemcpyDeviceToHost));
    cout << "\n The global atomicAdd reduction took :: " << 1.e3 * (t1e - t1) << " ms"
         << "\t \t"
         << "Sum = " << *h_ga_out << endl;
    CUDADEBUG(cudaDeviceReset());
    /******************************************************* Timing the shared atomicAdd kernel**/

    float *d_sa_in, *d_sa_out, *h_sa_out;
    CUDADEBUG(cudaMalloc((void **)&d_sa_in, bytes));
    CUDADEBUG(cudaMalloc((void **)&d_sa_out, sizeof(float)));
    CUDADEBUG(cudaMemcpy(d_sa_in, h_arr, bytes, cudaMemcpyHostToDevice));
    h_sa_out = (float *)malloc(sizeof(float));
    double t2 = omp_get_wtime();
    sum_reduction_shared_atomic<<<grid, block>>>(d_sa_in, d_sa_out, N / sizeof(float));
    CUDADEBUG(cudaDeviceSynchronize());
    double t2e = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_sa_out, d_sa_out, sizeof(float), cudaMemcpyDeviceToHost));
    cout << "\n The shared memory atomicAdd reduction took :: " << 1.e3 * (t2e - t2) << " ms"
         << "\t \t"
         << "Sum = " << *h_sa_out << endl;
    CUDADEBUG(cudaDeviceReset());

    /****************************************************** Timing the divergent reduction kernel**/

    float *d_da_in, *d_da_out, *h_da_out;
    CUDADEBUG(cudaMalloc((void **)&d_da_in, bytes));
    CUDADEBUG(cudaMalloc((void **)&d_da_out, bytes));
    CUDADEBUG(cudaMemcpy(d_da_in, h_arr, bytes, cudaMemcpyHostToDevice));
    h_da_out = (float *)malloc(bytes);
    double t3 = omp_get_wtime();
    sum_reduction_divergent<<<grid, block>>>(d_da_in, d_da_out);
    sum_reduction_divergent<<<1, block>>>(d_da_out, d_da_out);
    CUDADEBUG(cudaDeviceSynchronize());
    double t3e = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_da_out, d_da_out, bytes, cudaMemcpyDeviceToHost));
    cout << "\n The divergent harris reduction took :: " << 1.e3 * (t3e - t3) << " ms"
         << "\t \t"
         << "Sum = " << h_da_out[0] << endl;
    CUDADEBUG(cudaDeviceReset());

    /****************************************************** Timing the Bank conflect reduction kernel**/

    float *d_bca_in, *d_bca_out, *h_bca_out;
    CUDADEBUG(cudaMalloc((void **)&d_bca_in, bytes));
    CUDADEBUG(cudaMalloc((void **)&d_bca_out, bytes));
    CUDADEBUG(cudaMemcpy(d_bca_in, h_arr, bytes, cudaMemcpyHostToDevice));
    h_bca_out = (float *)malloc(bytes);
    double t4 = omp_get_wtime();
    sum_reduction_BCon<<<grid, block>>>(d_bca_in, d_bca_out);
    sum_reduction_BCon<<<1, block>>>(d_bca_out, d_bca_out);
    CUDADEBUG(cudaDeviceSynchronize());
    double t4e = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_bca_out, d_bca_out, bytes, cudaMemcpyDeviceToHost));
    cout << "\n The Bank conflect harris reduction took :: " << 1.e3 * (t4e - t4) << " ms"
         << "\t \t"
         << "Sum = " << h_bca_out[0] << endl;
    CUDADEBUG(cudaDeviceReset());

    /*********************************************** Timing the conflect free reduction kernel but with idle threads**/

    float *d_nbca_in, *d_nbca_out, *h_nbca_out;
    CUDADEBUG(cudaMalloc((void **)&d_nbca_in, bytes));
    CUDADEBUG(cudaMalloc((void **)&d_nbca_out, bytes));
    CUDADEBUG(cudaMemcpy(d_nbca_in, h_arr, bytes, cudaMemcpyHostToDevice));
    h_nbca_out = (float *)malloc(bytes);
    double t5 = omp_get_wtime();
    sum_reduction_NoBCon<<<grid, block>>>(d_nbca_in, d_nbca_out);
    sum_reduction_NoBCon<<<1, block>>>(d_nbca_out, d_nbca_out);
    CUDADEBUG(cudaDeviceSynchronize());
    double t5e = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_nbca_out, d_nbca_out, bytes, cudaMemcpyDeviceToHost));
    cout << "\n The Bank conflect free harris reduction with idle threads took :: " << 1.e3 * (t5e - t5) << " ms"
         << "\t \t"
         << "Sum = " << h_nbca_out[0] << endl;
    CUDADEBUG(cudaDeviceReset());

    /*********************************************** timing the harris reduction kernel with no idle threads in blocks**/

    float *d_nia_in, *d_nia_out, *h_nia_out;
    CUDADEBUG(cudaMalloc((void **)&d_nia_in, bytes));
    CUDADEBUG(cudaMalloc((void **)&d_nia_out, bytes));
    CUDADEBUG(cudaMemcpy(d_nia_in, h_arr, bytes, cudaMemcpyHostToDevice));
    h_nia_out = (float *)malloc(bytes);
    double t6 = omp_get_wtime();
    sum_reduction_Noidle<<<grididle, block>>>(d_nia_in, d_nia_out);
    sum_reduction_Noidle<<<1, block>>>(d_nia_out, d_nia_out);
    CUDADEBUG(cudaDeviceSynchronize());
    double t6e = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_nia_out, d_nia_out, bytes, cudaMemcpyDeviceToHost));
    cout << "\n The harris reduction with No idle threads in blocks took :: " << 1.e3 * (t6e - t6) << " ms"
         << "\t \t"
         << "Sum = " << h_nia_out[0] << endl;
    CUDADEBUG(cudaDeviceReset());

    /********************************************** Timing the Hariis reduction kernel with warpreduce ****/

    float *d_wa_in, *d_wa_out, *h_wa_out;
    CUDADEBUG(cudaMalloc((void **)&d_wa_in, bytes));
    CUDADEBUG(cudaMalloc((void **)&d_wa_out, bytes));
    CUDADEBUG(cudaMemcpy(d_wa_in, h_arr, bytes, cudaMemcpyHostToDevice));
    h_wa_out = (float *)malloc(bytes);
    double t7 = omp_get_wtime();
    sum_reduction_warp<<<grididle, block>>>(d_wa_in, d_wa_out);
    sum_reduction_warp<<<1, block>>>(d_wa_out, d_wa_out);
    CUDADEBUG(cudaDeviceSynchronize());
    double t7e = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_wa_out, d_wa_out, bytes, cudaMemcpyDeviceToHost));
    cout << "\n The harris reduction with warp reduce took :: " << 1.e3 * (t7e - t7) << " ms"
         << "\t \t"
         << "Sum = " << h_wa_out[0] << endl;
    CUDADEBUG(cudaDeviceReset());

    /********************************************** Timing the Haris reduction kernel with warp shuffle **/

    float *d_wsa_in, *d_wsa_out, *h_wsa_out;
    CUDADEBUG(cudaMalloc((void **)&d_wsa_in, bytes));
    CUDADEBUG(cudaMalloc((void **)&d_wsa_out, bytes));
    CUDADEBUG(cudaMemcpy(d_wsa_in, h_arr, bytes, cudaMemcpyHostToDevice));
    h_wsa_out = (float *)malloc(bytes);
    double t8 = omp_get_wtime();
    sum_reduction_shfl<<<grididle, block>>>(d_wsa_in, d_wsa_out, N);
    sum_reduction_shfl<<<1, block>>>(d_wsa_out, d_wsa_out, N);
    CUDADEBUG(cudaDeviceSynchronize());
    double t8e = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(h_wsa_out, d_wsa_out, bytes, cudaMemcpyDeviceToHost));
    cout << "\n The harris reduction with warp shuffle took :: " << 1.e3 * (t8e - t8) << " ms"
         << "\t \t"
         << "Sum = " << h_wsa_out[0] << endl;
    CUDADEBUG(cudaDeviceReset());

    /*****************************************************************************************************/

    cudaFree(d_bca_in);
    cudaFree(d_bca_out);
    cudaFree(d_da_in);
    cudaFree(d_da_out);
    cudaFree(d_ga_in);
    cudaFree(d_ga_out);
    cudaFree(d_nbca_in);
    cudaFree(d_nbca_out);
    cudaFree(d_nia_in);
    cudaFree(d_nia_out);
    cudaFree(d_sa_in);
    cudaFree(d_sa_out);
    cudaFree(d_wa_in);
    cudaFree(d_wa_out);
    cudaFree(d_wsa_in);
    cudaFree(d_wa_out);

    free(h_arr);
    free(h_bca_out);
    free(h_da_out);
    free(h_ga_out);
    free(h_nbca_out);
    free(h_nia_out);
    free(h_sa_out);
    free(h_wa_out);
    free(h_wsa_out);
}
