/*
standard matrix mult
*/


#include <iostream>
#include <math.h>


__global__ 
void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}

void matrixMultiplication(float *A, float *B, float *C, int N){

    // declare the number of blocks per grid and the number of threads 
    // per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 1024){
            threadsPerBlock.x = 1024;
            threadsPerBlock.y = 1;
            blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
        }
    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}

int main(int argc, char* argv[])
{

  if(argc < 2) std::cout << "Needs a dimension parameter.\n";
  int N = atoi(argv[1]);
  bool output = atoi(argv[2]);

  float* A;
  cudaError_t result = cudaMallocManaged(&A, N*N*sizeof(float));
  if( result != cudaSuccess)
  {
    throw std::runtime_error("Failed allocation.");
  }
  float* B;
  result = cudaMallocManaged(&B, N*N*sizeof(float));
  if( result != cudaSuccess)
  {
    throw std::runtime_error("Failed allocation.");
  }
  float* C;
  result = cudaMallocManaged(&C, N*N*sizeof(float));
  if( result != cudaSuccess)
  {
    throw std::runtime_error("Failed allocation.");
  }




  for(int i=0;  i < N*N; ++i)
  {
    A[i] = 1.2345;
    B[i] = 1.2345;
    C[i] = 0;
  }

  // if output set to 1, display A and B
  if(output)
  {
    for(int i = 0; i < N*N; ++i)
    {
      if (i%N == 0) std::cout << "\n";
      std::cout << A[i] << " ";
    }

    for(int i = 0; i < N*N; ++i)
    {
      if (i%N == 0) std::cout << "\n";
      std::cout << B[i] << " ";
    }
  }

  matrixMultiplication(A, B, C, N);
  //matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N); 
  // simpleMatMulKernell<<<1,256>>>(A,B,C,w);
  cudaDeviceSynchronize();

  // if output set to 1, show C after mult
  if(output)
  {
    for(int i =0; i < N*N; ++i)
    {
      if (i%N == 0) std::cout << "\n";
      std::cout << C[i] << " ";
    }
  }
  std::cout << "\nC[0] : " << C[0] << "\n";

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}

