#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cassert>

#define TILE_WIDTH 32 /* Tiling size of the Matrices for puting them into the shared memory of the Blocks*/
#define BLOCK_SIZE 32
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
    //cudaError_t cudaGetDeviceProperties(cudaDeviceProp & prop, int Device);
}
/*****************************************************************************************************/
// initialize a matrix with random floating point numbers
void initMatrix(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}
/*******************************************************************************************/
void print(float *A, const int size)
{
    cout << "\n"
         << endl;
    for (int i = 0; i < size; i++)
        cout << A[i] << ", ";
    cout << endl;
}
/********************************************************************************************/
/*The following is the CPU matrix multiplication function */
void cpu_MM(float *M, const int Mr, const int Mc, float *N, const int Nc, float *P)
{
    for (int i = 0; i < Mr; i++)
    {
        for (int j = 0; j < Nc; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < Mc; k++)
            {
                sum += M[i * Mc + k] * N[k * Nc + j];
            }
            P[i * Nc + j] = sum;
        }
    }
}
/*****************************************************************************************************/
// verify if two matrices match each other within the floating point tolerance
bool compareMatrices(float *A, float *B, int m, int n)
{
    const double epsilon = 1.0e-5;
    bool match = true;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int linInd = i * n + j;
            if (fabs(A[linInd] - B[linInd]) >= epsilon)
            {
                match = false;
                printf(" arrays do not match! \n");
                printf(" A:%e != B:%e at (i=%d,j=%d) \n", A[linInd], B[linInd], i, j);
                exit(-1);
            }
        }
    }
    printf(" arrays match! \n");
    return match;
}
/******************************************************************************************************/
/*The following is the kernel function for the vanila matrix multiplication kernel*/
__global__ void tiled_MM(const float *d_M, const float *d_N, float *d_P, const int width, const int MR, const int Nc)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float pvalue = 0.0;
    if (i < MR && j < Nc)
    {
        for (int k = 0; k < width; k++)
        {
            float Melement = d_M[i * width + k];
            float Nelement = d_N[k * Nc + j];
            pvalue += Melement * Nelement;
        }
        d_P[i * Nc + j] = pvalue;
    }
}
/*****************************************************************************************************/
/*The following is the kernel function for the tiled matrix multiplication using the shared memory */
__global__ void matMul_shared(float *d_M, float *d_N, float *d_P, int m, int k, int n)
{
    assert(blockDim.x == TILE_WIDTH && blockDim.y == TILE_WIDTH);
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0;
    //double numberoftiles =ceil(m/TILE_WIDTH);
    if (m == k == n) // if the matrices are squared
    {
        for (int l = 0; l < m / TILE_WIDTH; ++l)
        { //iterate through tiles
            for (int j = 0; j < TILE_WIDTH; ++j)
            { //iterate through elements in the tile
                sum = sum + d_M[(row * m) + (l * TILE_WIDTH + j)] * d_N[(l * TILE_WIDTH + j) * m + col];
            }
            __syncthreads();
        }
        d_P[row * m + col] = sum;
    }
    else // if the matrices  are not squared
    {
        for (int l = 0; l < ceil((float)k / TILE_WIDTH); ++l)
        { //iterate through tiles
            if (row < m && l * TILE_WIDTH + tx < k)
                ds_A[ty][tx] = d_M[row * k + l * TILE_WIDTH + tx];
            else
                ds_A[ty][tx] = 0.0;
            if (l * TILE_WIDTH + ty < k && col < n)
                ds_B[ty][tx] = d_N[(l * TILE_WIDTH + ty) * n + col];
            else
                ds_B[ty][tx] = 0.0;
            __syncthreads();
            for (int j = 0; j < TILE_WIDTH && j < k; ++j)
            { //iterate through elements in the tile
                sum = sum + ds_A[ty][j] * ds_B[j][tx];
            }
            __syncthreads();
        }
        if (row < m && col < n)
            d_P[row * n + col] = sum;
    }
}

/*************************************************************************************************/
/*************************************************************************************************/
int main(int argc, char **argv)
{
    //enquiring the execution configuration from User
    // int blockX = atoi(argv[1]);
    // int blockY = atoi((argv[2]));
    /* properties of the GPU device*/
    info();
    //size of the Matrices
    const int MR = atoi(argv[1]);
    const int MC = atoi(argv[2]);
    const int NR = atoi(argv[3]);
    const int NC = atoi(argv[4]);
    // original Host arrays. Note tsat there is two output pointer: d_P for CPU and gpu_P for output calculated by GPU
    float *h_M, *h_N, *h_P, *gpu_P;
    // allocating memory to the original matrices
    h_M = (float *)malloc(MR * MC * sizeof(float));
    h_N = (float *)malloc(NR * NC * sizeof(float));
    h_P = (float *)malloc(MR * NC * sizeof(float));
    gpu_P = (float *)malloc(MR * NC * sizeof(float));
    // initilizing the two operand matrices randomly
    initMatrix(h_M, MR * MC);
    initMatrix(h_N, NR * NC);

    /***************************** starting timing with CPU version of matrix Mult function*/
    double start1 = omp_get_wtime();
    cpu_MM(h_M, MR, MC, h_N, NC, h_P);
    double end1 = omp_get_wtime();
    cout << "\n The CPU version of the MM takes " << 1.e3 * (end1 - start1) << " ms" << endl;
    //print(h_P, MR * NC);
    /*############################ declaring and  allocating memory for device matrices */
    float *d_M, *d_N, *d_P;
    CUDADEBUG(cudaMalloc((void **)&d_M, MR * MC * sizeof(float)));
    CUDADEBUG(cudaMalloc((void **)&d_N, NR * NC * sizeof(float)));
    CUDADEBUG(cudaMalloc((void **)&d_P, MR * NC * sizeof(float)));
    //transfering the data from Host to DRAM
    CUDADEBUG(cudaMemcpy(d_M, h_M, MR * MC * sizeof(float), cudaMemcpyHostToDevice));
    CUDADEBUG(cudaMemcpy(d_N, h_N, NR * NC * sizeof(float), cudaMemcpyHostToDevice));
    //configuring the grid
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((NC + block.x - 1) / block.x, (MR + block.y - 1) / block.y);

    /******************************timing the vanila MM kernel execution */
    // double star2 = omp_get_wtime();
    // tiled_MM<<<grid, block>>>(d_M, d_N, d_P, NR, MR, NC);
    // cudaDeviceSynchronize();
    // double end2 = omp_get_wtime();
    // cout << "\n The vanila kernel took " << 1.e3 * (end2 - star2) << " ms" << endl;
    // //transfering the Data from GPU back to RAM
    // CUDADEBUG(cudaMemcpy(gpu_P, d_P, MR * NC * sizeof(float), cudaMemcpyDeviceToHost));
    // cudaDeviceReset();
    // // print(gpu_P, MR * NC);
    // //checking the accuracy of the results
    // compareMatrices(h_P, gpu_P, MR, NC);

    /******************************timing the shared tiled MM kernel execution */
    double star3 = omp_get_wtime();
    matMul_shared<<<grid, block>>>(d_M, d_N, d_P, MR, MC, NC);
    cudaDeviceSynchronize();
    double end3 = omp_get_wtime();
    cout << "\n The shared tiled kernel for nonsquared  took " << 1.e3 * (end3 - star3) << " ms"
         << "\n"
         << endl;
    //transfering the Data from GPU back to RAM
    CUDADEBUG(cudaMemcpy(gpu_P, d_P, MR * NC * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceReset();
    //print(gpu_P, MR * NC);
    //checking the accuracy of the results
    compareMatrices(h_P, gpu_P, MR, NC);

    /* freeeing the memories */
    free(h_M);
    free(h_N);
    free(h_P);
    free(gpu_P);
    CUDADEBUG(cudaFree(d_M));
    CUDADEBUG(cudaFree(d_N));
    CUDADEBUG(cudaFree(d_P));

    return 0;
}