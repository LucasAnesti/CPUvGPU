/*
isolating the kernel to compile using the -cubin option
which yields info about:
  - register per thread
  - shared memory per thread block
These are used in occupancy calculation

Output for this program:

ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z3addiPfS_' for 'sm_30'
ptxas info    : Function properties for _Z3addiPfS_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, 344 bytes cmem[0]
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



