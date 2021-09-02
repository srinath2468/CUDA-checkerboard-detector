#ifndef _CUDA_ERROR_CHECK_H_
#define _CUDA_ERROR_CHECK_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

// Usage --- use the function after calling the kernel
// ---------------------------------------------------
// kernelname<<<blocks, threads>>>(params..);
// cudaGetError("kernelname");
// ----------------------------------------------------

void inline cudaGetError(const char *kernelName)
{

  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "<<<CUDA>>> Error: %s  at kernel %s \n",
            cudaGetErrorString(err), kernelName);
    exit(-1);
  }
}

#endif