#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cassert>
#define TILE_WIDTH 16 // This Constant is used by Mult_Shared Kernel to define the Size of Shared memory Array
#define BlockX 16
#define BlockY 16
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
/*The following print out the information of the Device*/
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
/*The following Function proint the input matrix on the screen */
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
/****************************************************************************************/
/*The following is the CPU matrix multiplication function */
void cpu_MM(float *M, float *N, const int w, const int h, float *P)
{
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < h; k++)
            {
                sum += M[i * h + k] * N[k * w + j];
            }
            P[i * w + j] = sum;
        }
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
/*The following the kernel for Matrix Multiplication using Texture memory without ILP and Loop Unrolling*/
__global__ void mul_texture(float *output, int h1, int w1, int h2, int w2, int ss, cudaTextureObject_t texObjA, cudaTextureObject_t texObjB)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u1 = x / (float)w1; //col iter
    float v1 = y / (float)h1; //row iter

    float u2 = x / (float)w2; //col iter
    float v2 = y / (float)h2; //row iter

    if (x < w2 && y < h1)
    {
        float psum = 0.0;
        int pid = y * w2 + x;
        for (int k = 0; k < ss; k++)
        {
            psum += tex2D<float>(texObjA, u1 + k, v1) * tex2D<float>(texObjB, u2, v2 + k);
        }
        output[pid] = psum;
    }
}
/***********************************************************************************************/
/*The following is the Matrix Multiplication Kernel with Texture Memory and Loop unrolling */
__global__ void mul_texture_ilp(float *output, int h1, int w1, int h2, int w2, int ss, cudaTextureObject_t texObjA, cudaTextureObject_t texObjB)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u1 = x / (float)w1; //col iter
    float v1 = y / (float)h1; //row iter

    float u2 = x / (float)w2; //col iter
    float v2 = y / (float)h2; //row iter

    if (x < w2 && y < h1)
    {
        float psum = 0.0;
        int pid = y * w2 + x;
#pragma unroll 16
        for (int k = 0; k < ss; k++)
        {
            psum += tex2D<float>(texObjA, u1 + k, v1) * tex2D<float>(texObjB, u2, v2 + k);
        }
        output[pid] = psum;
    }
}

/*#############################################################################################*/
int main(int argc, char *argv[])
{
    const int n = atoi(argv[1]);
    /* setting the dimensions of the matrices */
    const int hig1 = 3 * n; //rows
    const int wid1 = 2 * n; // columns
    const int hig2 = 3 * n;
    const int wid2 = 2 * n;
    /*Setting the size of the matrices */
    const size_t size1 = hig1 * wid1 * sizeof(float);
    const size_t size2 = hig2 * wid2 * sizeof(float);

    float *h_A, *h_B;             //pointer to the matrices on the Host memory
    h_A = (float *)malloc(size1); // allocating memory to matrix A on Host
    h_B = (float *)malloc(size2); // allocating memory to matrix B on Host
    //filling the Matrices with  random numbers
    initMatrix(h_A, hig1 * wid1);
    initMatrix(h_B, hig2 * wid2);

    // Matrix Multiplication on CPU
    float *cpu_mult;
    const size_t size3 = hig1 * wid2 * sizeof(float);
    cpu_mult = (float *)malloc(size3);
    cpu_MM(h_A, h_B, hig1, wid2, cpu_mult);
    //print(cpu_mult, hig1 * wid2, wid2);
    cout << "\n **************************************" << endl;

    /* creating chanles for texture memory */
    cudaChannelFormatDesc channelDesc_A = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelDesc_B = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    /* declaring the matrices on the Cuda memory */
    cudaArray_t d_A, d_B;

    /*alocating memory on Device to CudaArrays A and B */
    CUDADEBUG(cudaMallocArray(&d_A, &channelDesc_A, hig1, wid1));
    CUDADEBUG(cudaMallocArray(&d_B, &channelDesc_B, hig2, wid2));

    /* Transfering the Data from Host to Device memory  */
    const size_t spitch1 = wid1 * sizeof(float);
    CUDADEBUG(cudaMemcpy2DToArray(d_A, 0, 0, h_A, spitch1, wid1 * sizeof(float), wid1, cudaMemcpyHostToDevice));
    const size_t spitch2 = wid2 * sizeof(float);
    CUDADEBUG(cudaMemcpy2DToArray(d_B, 0, 0, h_B, spitch2, wid2 * sizeof(float), wid2, cudaMemcpyHostToDevice));

    /*Setting The texture memory refrence parameters for Two matrices*/
    struct cudaResourceDesc resDescA;
    memset(&resDescA, 0, sizeof(resDescA));
    resDescA.resType = cudaResourceTypeArray;
    resDescA.res.array.array = d_A;
    struct cudaTextureDesc texDescA;
    memset(&texDescA, 0, sizeof(texDescA));
    texDescA.addressMode[0] = cudaAddressModeWrap;
    texDescA.addressMode[1] = cudaAddressModeWrap;
    texDescA.filterMode = cudaFilterModeLinear;
    texDescA.readMode = cudaReadModeElementType;
    texDescA.normalizedCoords = 1;
    struct cudaResourceDesc resDescB;
    memset(&resDescB, 0, sizeof(resDescB));
    resDescB.resType = cudaResourceTypeArray;
    resDescB.res.array.array = d_B;
    struct cudaTextureDesc texDescB;
    memset(&texDescB, 0, sizeof(texDescB));
    texDescB.addressMode[0] = cudaAddressModeWrap;
    texDescB.addressMode[1] = cudaAddressModeWrap;
    texDescB.filterMode = cudaFilterModeLinear;
    texDescB.readMode = cudaReadModeElementType;
    texDescB.normalizedCoords = 1;

    /*Create texture object*/
    cudaTextureObject_t texObjA = 0;
    cudaCreateTextureObject(&texObjA, &resDescA, &texDescA, NULL);
    cudaTextureObject_t texObjB = 0;
    cudaCreateTextureObject(&texObjB, &resDescB, &texDescB, NULL);

    /***********************************************/
    /** starting The matrix multiplication Test ****/
    /***********************************************/

    /* execution configuration */
    dim3 block(BlockX, BlockY);
    dim3 grid((wid2 + block.x - 1) / block.x, (hig1 + block.y - 1) / block.y);

    /*********Host pointer to the result of the Matrix Multiplication using mul_texture Krenel */
    float *h_C;
    h_C = (float *)malloc(size3);
    /*allocate device memory for result*/
    float *d_C;
    CUDADEBUG(cudaMalloc((void **)&d_C, size3));
    /*launching the matrix mult kernel with texture memory*/
    double s1 = omp_get_wtime();
    mul_texture<<<grid, block>>>(d_C, hig1, wid1, hig2, wid2, wid1, texObjA, texObjB);
    CUDADEBUG(cudaDeviceSynchronize());
    double e1 = omp_get_wtime();
    cout << "\n The Multiplication Kernel using Texture Memory Took " << 1.e3 * (e1 - s1) << "ms" << endl;
    CUDADEBUG(cudaMemcpy(h_C, d_C, size3, cudaMemcpyDeviceToHost));
    //print(h_C, hig1 * wid2, wid2);
    verify(cpu_mult, h_C, hig1, wid2);

    /***********Host pointer to the result of the Matrix Multiplication using Mul_texture_ilp kernel */
    float *h_c_i;
    h_c_i = (float *)malloc(size3);
    /*allocation device memory for results*/
    float *d_c_i;
    CUDADEBUG(cudaMalloc((void **)&d_c_i, size3));
    /*launching the matrix Multiplicatio kernel with texture memory and Unrolling loop */
    double s2 = omp_get_wtime();
    mul_texture_ilp<<<grid, block>>>(d_c_i, hig1, wid1, hig2, wid2, wid1, texObjA, texObjB);
    CUDADEBUG(cudaDeviceSynchronize());
    double e2 = omp_get_wtime();
    cout << "\n The Multiplication Kernel using Texture and loop unrolling Took " << 1.e3 * (e2 - s2) << "ms" << endl;
    CUDADEBUG(cudaMemcpy(h_c_i, d_c_i, size3, cudaMemcpyDeviceToHost));
    //print(h_c_i, hig1 * wid2, wid2);
    verify(cpu_mult, h_c_i, hig1, wid2);

    /************Host pointer to the result of the Matrix Mulptiplication kernel using shared memory */
    float *h_c_s;
    h_c_s = (float *)malloc(size3);
    float *d_C_s, *d_A_s, *d_B_s;
    CUDADEBUG(cudaMalloc((void **)&d_A_s, size1));
    CUDADEBUG(cudaMalloc((void **)&d_B_s, size2));
    CUDADEBUG(cudaMalloc((void **)&d_C_s, size3));
    CUDADEBUG(cudaMemcpy(d_A_s, h_A, size1, cudaMemcpyHostToDevice));
    CUDADEBUG(cudaMemcpy(d_B_s, h_B, size2, cudaMemcpyHostToDevice));
    double s3 = omp_get_wtime();
    matMul_shared<<<grid, block>>>(d_A_s, d_B_s, d_C_s, hig1, wid1, wid2);
    CUDADEBUG(cudaDeviceSynchronize());
    double e3 = omp_get_wtime();
    cout << "\n The Multiplication Kernel using shared Memory Took " << 1.e3 * (e3 - s3) << "ms" << endl;
    CUDADEBUG(cudaMemcpy(h_c_s, d_C_s, size3, cudaMemcpyDeviceToHost));
    verify(cpu_mult, h_c_s, hig1, wid2);

    /*resting teh divice and releasing the memories */
    CUDADEBUG(cudaDeviceReset());
    cudaFree(d_A_s);
    cudaFree(d_B_s);
    cudaFree(d_C);
    cudaFree(d_c_i);
    cudaFree(d_C_s);
    cudaFreeArray(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_c_i);
    free(h_c_s);
    return 0;
}
