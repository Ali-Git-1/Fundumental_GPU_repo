#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>

//======================================================================================
// matrix addition kernel
__global__ void cudaMatrixAdd(float *A, float *B, float *C, int m, int n)
{
	// global row index of the thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// global column index of the thread
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// check if the thread is accessing an element within the array boundaries
	if ((row < m) && (col < n))
	{
		// calculate global 1d (linear) index of the data element
		// (assuming one-to-one mapping between the thread and the data element)
		int linInd = row * n + col;
		C[linInd] = A[linInd] + B[linInd];
	}
}
//======================================================================================
// serial matrixAdd
void serialMatrixAdd(float *A, float *B, float *C, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int linInd = i * n + j;
			C[linInd] = A[linInd] + B[linInd];
		}
	}
}
//======================================================================================
// output cuda error
void cudaErr(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		printf(" cuda error in file '%s' in line %i : %s.",
			   __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
//=========================================================================================
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
//=======================================================================================
// verify if two matrices match each other within the floating point tolerance
bool compareMatrices(float *A, float *B, int m, int n)
{
	double epsilon = 1.0e-8;
	bool match = true;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int linInd = i * n + j;
			if (fabs(A[linInd] - B[linInd]) > epsilon)
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
//================================================================================================================

// first two arguments are the matrix dimensions, the last two arguments are the block dimensions
int main(int argc, char **argv)
{
	int h_m, h_n; // matrix dimensions
	int size;
	dim3 gridDimensions, blockDimensions;
	double start, end;

	if (argc == 5)
	{
		h_m = atoi(argv[1]);
		h_n = atoi(argv[2]);
		blockDimensions.x = atoi(argv[3]);
		blockDimensions.y = atoi(argv[4]);
		// we aren't using the third dimension, so set it to unity
		blockDimensions.z = 1;
		// calculate number of blocks needed to process the matrix
		gridDimensions.x = (h_n + blockDimensions.x - 1) / blockDimensions.x; // alternatively, one can use the ceil function
		gridDimensions.y = (h_m + blockDimensions.y - 1) / blockDimensions.y;
		gridDimensions.z = 1;
	}
	else
	{
		printf(" wrong number of input parameters \n");
		exit(-1);
	}

	size = h_m * h_n * sizeof(float);

	// declare and allocate the host matrices
	float *h_A, *h_B, *h_Cs, *h_Cp;
	h_A = (float *)malloc(size);
	h_B = (float *)malloc(size);
	h_Cs = (float *)malloc(size);
	h_Cp = (float *)malloc(size);

	// declare and allocate the device matrices
	float *d_A, *d_B, *d_C;
	cudaErr(cudaMalloc((void **)&d_A, size));
	cudaErr(cudaMalloc((void **)&d_B, size));
	cudaErr(cudaMalloc((void **)&d_C, size));

	// initialize the input matrices
	initMatrix(h_A, h_m * h_n);
	initMatrix(h_B, h_m * h_n);

	start = omp_get_wtime();

	// calculate the serial result for verification purposes
	serialMatrixAdd(h_A, h_B, h_Cs, h_m, h_n);

	end = omp_get_wtime();
	printf(" serial execution time=%e ms \n", 1.e3 * (end - start));

	// calculate the parallel result:

	// transfer the input matrices to device
	cudaErr(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	cudaErr(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	// execute the matrix addition kernel,
	// first time to let the system "warm up", second time to measure the execution time
	cudaMatrixAdd<<<gridDimensions, blockDimensions>>>(d_A, d_B, d_C, h_m, h_n);
	cudaErr(cudaDeviceSynchronize());

	start = omp_get_wtime();

	cudaMatrixAdd<<<gridDimensions, blockDimensions>>>(d_A, d_B, d_C, h_m, h_n);
	cudaErr(cudaDeviceSynchronize());

	end = omp_get_wtime();
	// note that it is just the kernel execution time,
	// to calculate the total parallel execution time,
	// you have to add the data transfer time
	printf(" parallel execution time=%e ms \n", 1.e3 * (end - start));

	// transfer the result to host
	cudaErr(cudaMemcpy(h_Cp, d_C, size, cudaMemcpyDeviceToHost));

	// verify that the matrices match
	compareMatrices(h_Cs, h_Cp, h_m, h_n);

	// clean up after yourself
	free(h_A);
	free(h_B);
	free(h_Cs);
	free(h_Cp);
	cudaErr(cudaFree(d_A));
	cudaErr(cudaFree(d_B));
	cudaErr(cudaFree(d_C));

	// release the device
	cudaDeviceReset();

	// return the success code
	return 0;
}