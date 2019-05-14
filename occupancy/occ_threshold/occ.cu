/*
Terminal input n is used as 2^n to determine input size
This program is used to time performance of input and 
demonstrate graphically occupancy as it relates to performance vs input size.
*/

#include <iostream>
#include <math.h>





//CUDA kernel to add elements of the matrix
// __global__ converts a function into a CUDA kernel
__global__
void add(int n, float *x, float *y)
{
  // index of the current thread within the block
  int index =  blockIdx.x * blockDim.x + threadIdx.x;
  // number of threads in a block
  int stride = blockDim.x * gridDim.x;

  // run each addition on a separate thread
  for (int i = index; i < n; i+=stride)
      y[i] = x[i] + y[i];


}









int main(int argc, char* argv[])
{
  if(argc < 1)
  {
    std::cout << "Use:  ./occ n \n     where n = 2^n as input size" << "\n";
  }
  
  int n = atoi(argv[1]);
  int N = 1<<n; // 2^n elements

  // Memory allocation in CUDA is done with cudaMallocManaged( , )
  float *x; float *y;
  cudaMallocManaged( &x, N*sizeof(float) );
  cudaMallocManaged( &y, N*sizeof(float) );


  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Launch the 'add' kernel, which invokes it in the GPU
  // Run kernel on 1M elements on the CPU
  // CHange the execution config to use 256 threads
  // dim3 threadsPerBlock(16,16);
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize; 
  // std::cout << "NumBlocks = " << numBlocks << "\n";
  add<<<numBlocks,blockSize>>>(N, x, y);

  // Wait for the GPU to synchronize before accessign through host(CPU)
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  /*float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
  */

  // Deallocating memory using cudaFree()
  cudaFree(x);
  cudaFree(y);

  return 0;
}

