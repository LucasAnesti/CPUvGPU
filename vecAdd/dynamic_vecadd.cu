
/*******************************
1 - Install nvidia-cuda-toolkit

2 - Compile this program using:

     nvcc add.cu -o add_cuda.out
*******************************/

/*
Program that runs block size dynamically. As the number of threads increase,
the number of blocks is determined as a function of threads and input size.
This provides a constant optimal performance even though the number of threads 
change 
*/

#include <iostream>
#include <math.h>
#include <ctime>
#include <cstdio>



//CUDA kernel to add elements of the matrix
// __global__ converts a function into a CUDA kernel
__global__
void add(int n, float *x, float *y)
{
  // index of the current thread within the block
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // number of threads in a block
  int stride = blockDim.x * gridDim.x;

  // run each addition on a separate thread
  for (int i = index; i < n; i+=stride)
      y[i] = x[i] + y[i];
}




int main(void)
{
  for(int t = 32; t <= 1024; t+=32)
  {
    int N = 1<<24; // 2^24 elements

    // Memory allocation in CUDA is done with cudaMallocManaged( , )
    float *x; float *y;
    cudaMallocManaged( &x, N*sizeof(float) );
    cudaMallocManaged( &y, N*sizeof(float) );


    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }


    std::clock_t start = clock();
    // Launch the 'add' kernel, which invokes it in the GPU
    int blockSize = t;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::cout << "BlockSize = " << t << ",NumBlocks = " << numBlocks << "\n";
    add<<<numBlocks,blockSize>>>(N, x, y);
    

    // Wait for the GPU to synchronize before accessign through host(CPU)
    cudaDeviceSynchronize();

    std::clock_t stop = clock();
    int duration = 1000 * (stop - start) / (double)CLOCKS_PER_SEC;
    //std::cout << "Running time using " << t << " threads = " << duration << "\n"; 
    std::cout << duration << "\n";

    // Check for errors (all values should be 3.0f)
    /*float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    */

    // Deallocating memory using cudaFree()
    cudaFree(x);
    cudaFree(y);
  }

  return 0;
}

