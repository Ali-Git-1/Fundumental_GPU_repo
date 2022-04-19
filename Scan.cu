#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cassert>
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK 1024
using namespace std;

/********************************************************************************************/
/*
The following macro is for cheking CUDA Errors while allocationg memory in Device and transfering the data
*/
#define DEBUG(str) std::cerr << "\033[1;37m" << __FILE__ << ":" << __LINE__ << ": \033[1;31merror:\033[0m " << str << std::endl;

#define CUDADEBUG(call)                     \
    {                                       \
        const cudaError_t err = call;       \
        if (err != cudaSuccess)             \
            DEBUG(cudaGetErrorString(err)); \
    }
/****************************************************************************************************/
void info(); /* This function prints out the properties of the GPU device*/

void initArray(int *ip, int size, bool fill_zero); /* initilizes an Array of given size with random elemnts*/

void print(int *A, const int size); /*printer of an array*/

void scan_cpu(int *out, int *in, int length); /* CPU version of the prefix Sum (scan) algorithm*/

int nextPowerOfTwo(int x); /* an axilary function for the kenels to find out wheter an arrays size is between tow consecutive power 2 numbers  */

bool check_results(int *host_ref, int *gpu_ref, const int n); /*verfiying the results of the GPU with CPU*/

/*.................................................................................................*/
void scanSmallDeviceArray(int *d_out, int *d_in, int length); /* Wrraper function for using paralle prefix sum algorithm on small size of array */
/* wrapper function to use kernels of scan algorithmn with large input arrays */
void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length);
void scanLargeDeviceArray(int *d_out, int *d_in, int length);
/* wrpper function to use in condut the test*/
float scan(int *output, int *input, int length);
void test(int data_size);
/*...................................................................................................*/
/*kernel to use scan Algorithm for arbitrary sizes in two forms for large and small arrays */
__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo);
__global__ void prescan_large(int *output, int *input, int n, int *sums);
__global__ void add(int *output, int length, int *n);
__global__ void add(int *output, int length, int *n1, int *n2);

/****************************************************************************************************/
/***************************************************************************************************/
/***************************************************************************************************/
int main()
{
    info();
    cout << endl;
    cout << "********************************************************************" << endl;
    const int max_data_size = 1000000;
    const int data_size_step = 100000;
    int data_size = 100000;
    for (; data_size <= max_data_size; data_size += data_size_step)
    {
        test(data_size);
        cout << "********************************************************************" << endl;
    }
    cout << endl;
}

/****************************************************************************************************/
void test(int data_size)
{
    const int size = data_size * sizeof(int);
    int *h_in, *h_out, *gpu_out;
    h_in = (int *)malloc(size);
    h_out = (int *)malloc(size);
    gpu_out = (int *)malloc(size);

    /*executing and timing of the CPU version scan*/
    initArray(h_in, data_size, false);
    double s = omp_get_wtime();
    scan_cpu(h_out, h_in, data_size);
    double e = omp_get_wtime();
    cout << " \t \t "
         << "Array size"
         << "\t "
         << "Runtime" << endl;
    cout << " CPU ::"
         << "\t \t" << data_size << "\t \t " << 1.e3 * (e - s) << " ms" << endl;

    /*executing and timing of the optimized paralle prefix sum */
    float gputi = scan(gpu_out, h_in, data_size);
    cout << " GPU ::"
         << "\t \t" << data_size << "\t \t " << gputi << " ms" << endl;

    /*variying the resulting arrays*/
    //check_results(h_out, gpu_out, data_size);
    /*demonstrating the result of the sum from CPU and from GPU */
    cout << " GPU_sum = " << gpu_out[data_size - 1] << "\t \t CPU_sum =  " << h_out[data_size - 1] << endl;
}
/***************************************************************************************************/
float scan(int *output, int *input, int length)
{
    int *d_out, *d_in;
    const int arraySize = length * sizeof(int);
    cudaMalloc((void **)&d_out, arraySize);
    cudaMalloc((void **)&d_in, arraySize);
    cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);
    // start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (length > ELEMENTS_PER_BLOCK)
    {
        scanLargeDeviceArray(d_out, d_in, length);
    }
    else
    {
        scanSmallDeviceArray(d_out, d_in, length);
    }
    // end timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}
/***************************************************************************************************/
void scanSmallDeviceArray(int *d_out, int *d_in, int length)
{
    int powerOfTwo = nextPowerOfTwo(length);
    prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo);
}
/***************************************************************************************************/
void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length)
{
    const int blocks = length / ELEMENTS_PER_BLOCK;
    const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);
    int *d_sums, *d_incr;
    cudaMalloc((void **)&d_sums, blocks * sizeof(int));
    cudaMalloc((void **)&d_incr, blocks * sizeof(int));

    prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

    const int sumsArrThreadsNeeded = (blocks + 1) / 2;
    if (sumsArrThreadsNeeded > THREADS_PER_BLOCK)
    {
        // perform a large scan on the sums arr
        scanLargeDeviceArray(d_incr, d_sums, blocks);
    }
    else
    {
        // only need one block to scan sums arr so can use small scan
        scanSmallDeviceArray(d_incr, d_sums, blocks);
    }
    add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);
    cudaFree(d_sums);
    cudaFree(d_incr);
}

/***************************************************************************************************/
void scanLargeDeviceArray(int *d_out, int *d_in, int length)
{
    int remainder = length % (ELEMENTS_PER_BLOCK);
    if (remainder == 0)
    {
        scanLargeEvenDeviceArray(d_out, d_in, length);
    }
    else
    {
        // perform a large scan on a compatible multiple of elements
        int lengthMultiple = length - remainder;
        scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple);
        // scan the remaining elements and add the (inclusive) last element of the large scan to this
        int *startOfOutputArray = &(d_out[lengthMultiple]);
        scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);
        add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
    }
}

/****************************************************************************************************/
/** this function prints the information of the GPU on the screeen */
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
/* initialize an Array with random intiger numbers*/
void initArray(int *ip, int size, bool fill_zero)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        if (!fill_zero)
            ip[i] = (rand() & 0xFF) / 10;
        else if (fill_zero)
            ip[i] = 0;
    }
}
/****************************************************************************************************/
void print(int *A, const int size)
{
    cout << "\n"
         << endl;
    for (int i = 0; i < size; i++)
        cout << A[i] << ", ";
    cout << endl;
}
/**************************************************************************************************/
/* The cpu version of th parallel prefix sum */
void scan_cpu(int *out, int *in, int length)
{
    out[0] = in[0];
    for (int i = 1; i < length; ++i)
        out[i] = in[i] + out[i - 1];
}
/**************************************************************************************************/
int nextPowerOfTwo(int x)
{
    int power = 1;
    while (power < x)
    {
        power *= 2;
    }
    return power;
}
/***************************************************************************************************/
/* verifying yhe correctness of the GPU result eith the results  of the cpu */
bool check_results(int *host_ref, int *gpu_ref, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (host_ref[i] != gpu_ref[i])
        {
            printf("WARNING: arrays do not match! "
                   "[%d] host: %d\tgpu: %d\n",
                   i, host_ref[i], gpu_ref[i]);
            return false;
        }
    }
    return true;
}
/***************************************************************************************************/
__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo)
{
    extern __shared__ int temp[]; // allocated on invocation
    int threadID = threadIdx.x;
    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    if (threadID < n)
    {
        temp[ai + bankOffsetA] = input[ai];
        temp[bi + bankOffsetB] = input[bi];
    }
    else
    {
        temp[ai + bankOffsetA] = 0;
        temp[bi + bankOffsetB] = 0;
    }
    int offset = 1;
    for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (threadID == 0)
    {
        temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
    }
    for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    if (threadID < n)
    {
        output[ai] = temp[ai + bankOffsetA];
        output[bi] = temp[bi + bankOffsetB];
    }
}
/****************************************************************************************************/
/****************************************************************************************************/
__global__ void prescan_large(int *output, int *input, int n, int *sums)
{
    extern __shared__ int temp[];
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * n;
    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = input[blockOffset + ai];
    temp[bi + bankOffsetB] = input[blockOffset + bi];
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    if (threadID == 0)
    {
        sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    output[blockOffset + ai] = temp[ai + bankOffsetA];
    output[blockOffset + bi] = temp[bi + bankOffsetB];
}
/***************************************************************************************************/
/****************************************************************************************************/
__global__ void add(int *output, int length, int *n)
{
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n[blockID];
}
/******************************************************************************************************/
__global__ void add(int *output, int length, int *n1, int *n2)
{
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}
/*******************************************************************************************************/
