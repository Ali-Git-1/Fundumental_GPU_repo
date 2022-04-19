#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cassert>
#include <math.h>
using namespace std;
#define TILE_DIM 32
#define BLOCK_ROWS 8
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
    printf("**************************************************************");
}
/*****************************************************************************************************/
// initialize a Tensor with random floating point numbers
void initMatrix(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}
/*****************************************************************************************************/
void print(float *A, const int size, const int csize)
{
    cout << "\n";
    for (int i = 0; i < size; i++)
    {
        if (i % csize == 0)
            cout << "\n"
                 << endl;
        cout << A[i] << ", ";
    }
    cout << endl;
}
/***************************************************************************************************/
/*The folowing verifies whether tow given matrixes are equal*/
bool verify(float *A, float *B, int m, int n)
{
    const double epsilon = 1.0e-8;
    bool match = true;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int linInd = i * n + j;
            if (fabs(A[linInd] - B[linInd]) >= epsilon)
            {
                match = false;
                printf("!!!!!!!!! arrays do not match !!!!!!!!!!! \n");
                printf(" A:%e != B:%e at (i=%d,j=%d) \n", A[linInd], B[linInd], i, j);
                exit(-1);
            }
        }
    }
    printf(" arrays match! \n");
    return match;
}
/*****************************************************************************************************/
/*cpu version of Matrix Transpose */
void transpose(float *mat, float *trans, const int mr, const int mc)
{
    //loop over transpose and assign the corrisponding matrix element to its elements
    for (int i = 0; i < mc; i++)
        for (int j = 0; j < mr; j++)
            trans[i * mr + j] = mat[j * mc + i];
}
/*****************************************************************************************************/
/*naive Row to Row copy kernel */
__global__ void Rowcopy(float *odata, const float *idata, const int ny, const int nx)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex < nx && yIndex < ny)
        odata[yIndex * nx + xIndex] = idata[yIndex * nx + xIndex];
}
/*****************************************************************************************************/
__global__ void copyTiled(float *odata, float *idata, int ny, int nx)
{
    // int x = blockIdx.x * TILE_DIM + threadIdx.x;
    // int y = blockIdx.y * TILE_DIM + threadIdx.y;
    // int width = gridDim.x * TILE_DIM;
    // for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    //     odata[(y + j) * width + x] = idata[(y + j) * width + x];
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index = xIndex + nx * yIndex;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[index + i * nx] = idata[index + i * nx];
    }
}
/*****************************************************************************************************/
/*Naive matrix transwpose kernel in Row major order */
__global__ void TransRowNaive(float *odata, const float *idata, const int ny, const int nx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny)
        odata[x * ny + y] = idata[y * nx + x];
}
/*****************************************************************************************************/
/*Naive matrix transpose kernel in Column major order*/
__global__ void TransColNaive(float *odata, const float *idata, const int ny, const int nx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny)
        odata[y * nx + x] = idata[x * ny + y];
}
/*****************************************************************************************************/
/* Tiled Naive  Matrix Transpose kernel */
__global__ void TiledTranspose(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
}
/******************************************************************************************************/
/*Shared Matrix transpose Kernel */
__global__ void transposeCoalesced(float *odata, float *idata, int width, int height, int nreps)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for (int r = 0; r < nreps; r++)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        __syncthreads();
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y];
    }
}
/***************************************************************************************************/
// __global__ void Coalesedtranspose(float *odata, float *idata, int width, int height)
// {
//     __shared__ float block[TILE_DIM][TILE_DIM + 1];
//     unsigned int xIndex, yIndex, index_in, index_out;

//     /* read the matrix tile into shared memory */
//     xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
//     yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
//     if ((xIndex < width) && (yIndex < height))
//     {
//         index_in = yIndex * width + xIndex;
//         block[threadIdx.y][threadIdx.x] = idata[index_in];
//     }
//     __syncthreads();
//     /* write the transposed matrix tile to global memory */
//     xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
//     yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
//     if ((xIndex < height) && (yIndex < width))
//     {
//         index_out = yIndex * height + xIndex;
//         odata[index_out] = block[threadIdx.x][threadIdx.y];
//     }
// }
/*##########################################################################################################*/
/*##########################################################################################################*/
int main(int argc, char **argv)
{
    info();
    const int nx = 4096; //columns
    const int ny = 4096; //Rows

    size_t Bytes = nx * ny * sizeof(float);
    // declaring the matrix pointer and its transpoze pointer
    float *matrix, *trans;
    // allocating memory to matrix and its transpose
    matrix = (float *)malloc(Bytes);
    trans = (float *)malloc(Bytes);
    //initializing the matrix randomly
    initMatrix(matrix, nx * ny);
    //print(matrix, RowSize * ColSize, ColSize);

    /*********************************** Timing the CPU version of Transapose Function **************/
    double cpustart = omp_get_wtime();
    transpose(matrix, trans, ny, nx);
    double cpuend = omp_get_wtime();
    //print(trans, RowSize * ColSize, RowSize);
    cout << "\n# Matrix Treanspose Operation on CPU took " << 1.e3 * (cpuend - cpustart) << "ms " << endl;

    /*********************************** Timing the simple Row_to_Row matrix copy kernel ***********/
    //declaring and allocation memory to the GPU output pointers
    float *g_c_M = (float *)malloc(Bytes);
    //declaring device pointer
    float *d_matrix, *d_copy;
    CUDADEBUG(cudaMalloc((void **)&d_matrix, Bytes));
    CUDADEBUG(cudaMalloc((void **)&d_copy, Bytes));
    //transfering the host matrix to device matrix
    CUDADEBUG(cudaMemcpy(d_matrix, matrix, Bytes, cudaMemcpyHostToDevice));
    // execution configuration
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(ceil((nx + TILE_DIM - 1) / TILE_DIM), ceil((ny + TILE_DIM - 1) / TILE_DIM));
    //launching the simple Row_to_Row copy kernel
    double s1 = omp_get_wtime();
    copyTiled<<<grid, block>>>(d_copy, d_matrix, ny, nx);
    cudaDeviceSynchronize();
    double e1 = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(g_c_M, d_copy, Bytes, cudaMemcpyDeviceToHost));
    // measuering the effective Bandwidth
    float bandwithCopy = ((nx * ny * 2 * sizeof(float))) / (1.e9 * (e1 - s1));
    cout << "\n# Row_To_Row Tiled Matrix Copy kernel took  " << 1.e3 * (e1 - s1) << " ms and The effective Bandwidth is " << bandwithCopy << endl;
    //print(g_c_M, nx * ny, nx);
    verify(matrix, g_c_M, ny, nx);

    /************************************* Timing the naive row major Transpos kernel **********************/
    //declaring and allocation memory to the GPU output transpose matrix
    float *g_T_M = (float *)malloc(Bytes);
    //declaring device pointer
    float *d_matr, *d_transp;
    CUDADEBUG(cudaMalloc((void **)&d_matr, Bytes));
    CUDADEBUG(cudaMalloc((void **)&d_transp, Bytes));
    //transfering the host matrix to device matrix
    CUDADEBUG(cudaMemcpy(d_matr, matrix, Bytes, cudaMemcpyHostToDevice));
    // execution configuration
    dim3 block1(TILE_DIM, TILE_DIM);
    dim3 grid1(ceil((nx + TILE_DIM - 1) / TILE_DIM), ceil((ny + TILE_DIM - 1) / TILE_DIM));
    //launching the naive transpose kernel
    double s2 = omp_get_wtime();
    TransRowNaive<<<grid1, block1>>>(d_transp, d_matr, ny, nx);
    cudaDeviceSynchronize();
    double e2 = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(g_T_M, d_transp, Bytes, cudaMemcpyDeviceToHost));
    // measuering the effective Bandwidth
    float bandwithNaiveTrans = ((ny * nx * 2 * sizeof(float))) / (1.e9 * (e2 - s2));
    cout << "\n# Naive matrix Transpose kernel in Row major  took  " << 1.e3 * (e2 - s2) << " ms and the effective bandwidth is " << bandwithNaiveTrans << endl;
    //print(g_T_M, ny * nx, ny);
    verify(trans, g_T_M, nx, ny);

    /************************************* Timing The naive Column major Transpose Kernel  ********************/
    //declaring and allocation memory to the GPU output transpose matrix
    //declaring device pointer
    float *g_T_MC = (float *)malloc(Bytes);
    float *d_matrC, *d_transpC;
    CUDADEBUG(cudaMalloc((void **)&d_matrC, Bytes));
    CUDADEBUG(cudaMalloc((void **)&d_transpC, Bytes));
    //transfering the host matrix to device matrix
    CUDADEBUG(cudaMemcpy(d_matrC, matrix, Bytes, cudaMemcpyHostToDevice));
    // execution configuration
    dim3 block2(TILE_DIM, TILE_DIM);
    dim3 grid2(ceil((nx + TILE_DIM - 1) / TILE_DIM), ceil((ny + TILE_DIM - 1) / TILE_DIM));
    //launching the naive transpose kernel
    double s3 = omp_get_wtime();
    TransColNaive<<<grid2, block2>>>(d_transpC, d_matrC, ny, nx);
    cudaDeviceSynchronize();
    double e3 = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(g_T_MC, d_transpC, Bytes, cudaMemcpyDeviceToHost));
    // measuering the effective Bandwidth
    float bandwithNaiveTransC = ((ny * nx * 2 * sizeof(float))) / (1.e9 * (e3 - s3));
    cout << "\n# Naive matrix Transpose kernel in Col major took  " << 1.e3 * (e3 - s3) << " ms and the effective bandwidth is " << bandwithNaiveTransC << endl;
    //print(g_T_MC, ny * nx, ny);
    verify(trans, g_T_MC, nx, ny);

    /**************************************** Timing Tiled matrix Transpose kernel ******************************/
    //declaring and allocation memory to the GPU output transpose matrix
    float *g_T_MT = (float *)malloc(Bytes);
    //declaring device pointer
    float *d_matrT, *d_transpT;
    CUDADEBUG(cudaMalloc((void **)&d_matrT, Bytes));
    CUDADEBUG(cudaMalloc((void **)&d_transpT, Bytes));
    //transfering the host matrix to device matrix
    CUDADEBUG(cudaMemcpy(d_matrT, matrix, Bytes, cudaMemcpyHostToDevice));
    // execution configuration
    dim3 block3(TILE_DIM, BLOCK_ROWS);
    dim3 grid3(ceil((nx + TILE_DIM - 1) / TILE_DIM), ceil((ny + TILE_DIM - 1) / TILE_DIM));
    //launching the naive transpose kernel
    double s4 = omp_get_wtime();
    TiledTranspose<<<grid3, block3>>>(d_transpT, d_matrT);
    cudaDeviceSynchronize();
    double e4 = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(g_T_MT, d_transpT, Bytes, cudaMemcpyDeviceToHost));
    // measuering the effective Bandwidth
    float bandwithNaiveTransT = ((ny * nx * 2 * sizeof(float))) / (1.e9 * (e4 - s4));
    cout << "\n# Naive Tiled matrix Transpose Kernel took  " << 1.e3 * (e4 - s4) << " ms and the effective bandwidth is " << bandwithNaiveTransT << endl;
    //print(g_T_MT, ny * nx, ny);
    verify(trans, g_T_MT, nx, ny);

    /************************************* Timing Shared Matrix Transpose Kernel************************/
    //declaring and allocation memory to the GPU output transpose matrix
    float *g_T_MS = (float *)malloc(Bytes);
    //declaring device pointer
    float *d_matrS, *d_transpS;
    CUDADEBUG(cudaMalloc((void **)&d_matrS, Bytes));
    CUDADEBUG(cudaMalloc((void **)&d_transpS, Bytes));
    //transfering the host matrix to device matrix
    CUDADEBUG(cudaMemcpy(d_matrS, matrix, Bytes, cudaMemcpyHostToDevice));
    // execution configuration
    dim3 block4(TILE_DIM, BLOCK_ROWS);
    dim3 grid4(ceil((nx + TILE_DIM - 1) / TILE_DIM), ceil((ny + TILE_DIM - 1) / TILE_DIM));
    //launching the naive transpose kernel
    double s5 = omp_get_wtime();
    transposeCoalesced<<<grid4, block4>>>(d_transpT, d_matrT, nx, ny, 1);
    cudaDeviceSynchronize();
    double e5 = omp_get_wtime();
    CUDADEBUG(cudaMemcpy(g_T_MS, d_transpS, Bytes, cudaMemcpyDeviceToHost));
    // measuering the effective Bandwidth
    float bandwithNaiveTransS = ((ny * nx * 2 * sizeof(float))) / (1.e9 * (e5 - s5));
    cout << "\n# Coolesed matrix Transpose Kernel with no Bank Conflict took  " << 1.e3 * (e5 - s5) / 1 << " ms and the effective bandwidth is " << bandwithNaiveTransS << endl;
    //print(g_T_MS, ny * nx, ny);
    verify(trans, g_T_MS, nx, ny);
}