cat matrixMult.cpp
#include <iostream>
#include <cmath>
#include <cstdlib>


void matrixMult( float* A, float* B, float* C, int N)
{
  for( int r = 0; r < N; ++r)
  {
    for( int c = 0; c < N; ++c)
    {
      float sum = 0;
      for(int k=0; k < N; k++) {
        sum += A[r * N + k] * B[k * N + c];
      }
      C[r * N + c] = sum;
    }
  }
}






int main(int argc, char* argv[])
{

  if(argc < 2) std::cout << "Needs a dimension parameter.\n";
  int N = atoi(argv[1]);
  bool output = atoi(argv[2]);
  // dimension = #row = #col since N is a square matrix
  int dimension = sqrt(N);

  float* A = new float[N*N];
  float* B = new float[N*N];
  float* C = new float[N*N];

  for(int i=0;  i < N*N; ++i)
  {
    A[i] = 1.2345;
    B[i] = 1.2345;
    C[i] = 0.0;
  }

  // if output parameter is 1 display A and B matrices
  if(output)
  {  for(int i =0; i < N*N; ++i)
     {
       if (i%N == 0) std::cout << "\n";
       std::cout << A[i] << " ";
     }

     for(int i =0; i < N*N; ++i)
     {
       if (i%N == 0) std::cout << "\n";
       std::cout << B[i] << " ";
     }
  }

  matrixMult(A,B,C,N);


  // if output parameter is 1 display C after mult
  if(output)
  {
    for(int i =0; i < N*N; ++i)
    {
      if (i%N == 0) std::cout << "\n";
      std::cout << C[i] << " ";
    }
  }
  std::cout << "\nC[0] : " << C[0] << "\n";



  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}

