#include "checkerboard/cuda/hessian.h"

__global__ void GaussianKernel(HessianParams params, byte *image,
                               float *gaussian) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > params.threads)
    return;

  int Iy = tid / params.resolution.x; // Y index of Matrix/GPU
  int Ix = tid % params.resolution.x; // X index of Matrix/GPU

  // Gaussian Image
  float gauss = 0;

  if (Ix < 1 || Ix > params.resolution.x - 2 || Iy < 1 ||
      Iy > params.resolution.y - 2)
    return;

  for (int i = 0; i < 9; i++) {

    int x = offsets[i][0];
    int y = offsets[i][1];

    int neighbourId = (((Iy + y) * params.resolution.x) + (Ix + x));

    float grayscale = (float)image[neighbourId];

    // float grayscale = image[neighbourId];

    gauss += (grayscale * gaussianKernel[i]);
  }

  if (gauss > 255) {
    gaussian[tid] = 255.0f;
  } else if (gauss < 0) {
    gaussian[tid] = 0.0f;
  } else {

    gaussian[tid] = (int)gauss;
  }
}

__global__ void FirstOrderDerivativeKernel(HessianParams params, float *image,
                                           float *xx, float *xy, float *yy,
                                           float *firstOrderDerivativeX,
                                           float *firstOrderDerivativeY) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > params.threads)
    return;

  // X and Y coordinates
  int Iy = floorf(tid / params.resolution.x); // Y index of Matrix
  int Ix = tid % params.resolution.x;         // X index of Matrix

  // finding Sobel and covariance
  float sobelX = 0, sobelY = 0;

  for (int i = 0; i < 49; i++) {

    int x = offsets7x7[i][0];
    int y = offsets7x7[i][1];

    int id = ((Iy + y) * params.resolution.x) + (Ix + x);

    if (Iy + y < 0 || (Iy + y) > params.resolution.y - 1 || Ix + x < 0 ||
        Ix + x > params.resolution.x - 1)
      continue;

    sobelX += (int)image[id] * sobel7x7[i][0];
    sobelY += (int)image[id] * sobel7x7[i][1];
  }

  firstOrderDerivativeX[tid] = sobelX;
  firstOrderDerivativeY[tid] = sobelY;
}

__global__ void HessianKernel(HessianParams params, float *xx, float *xy,
                              float *yy, float *firstOrderDerivativeX,
                              float *firstOrderDerivativeY, float *eigenValue1,
                              float *eigenValue2, int *maxCornerValue) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > params.threads)
    return;

  // X and Y coordinates
  int Iy = floorf(tid / params.resolution.x); // Y index of Matrix
  int Ix = tid % params.resolution.x;         // X index of Matrix

  if (Ix < 3 || Ix > params.resolution.x - 4 || Iy < 3 ||
      Iy > params.resolution.y - 4)
    return;

  float sobelXX = 0, sobelYY = 0, sobelXY = 0;
  for (int i = 0; i < 49; i++) {

    int x = offsets7x7[i][0];
    int y = offsets7x7[i][1];

    int id = ((Iy + y) * params.resolution.x) + (Ix + x);

    if (Iy + y < 0 || (Iy + y) > params.resolution.y - 1 || Ix + x < 0 ||
        Ix + x > params.resolution.x - 1)
      continue;

    sobelXX += firstOrderDerivativeX[id] * sobel7x7[i][0];
    sobelYY += firstOrderDerivativeY[id] * sobel7x7[i][1];
    sobelXY += firstOrderDerivativeX[id] * sobel7x7[i][1];
  }

  xx[tid] = (int)sobelXX;
  xy[tid] = (int)sobelXY;
  yy[tid] = (int)sobelYY;

  float covAA = xx[tid];
  float covAB = xy[tid];
  float covBB = yy[tid];

  eigenValue1[tid] =
      0.5f *
      ((covAA + covBB) + sqrtf(powf(covAA - covBB, 2) + (4 * powf(covAB, 2))));
  eigenValue2[tid] =
      0.5f *
      ((covAA + covBB) - sqrtf(powf(covAA - covBB, 2) + (4 * powf(covAB, 2))));

  float det = eigenValue2[tid] * eigenValue1[tid];

  int cornerValue = (float)(abs(det)) / 100000;
  atomicMax(maxCornerValue, cornerValue);
}

__global__ void GetCornerKernel(HessianParams params, float *eigenValue1,
                                float *eigenValue2, int *maxCornerValue,
                                int *unfilteredCount, int *unfilteredLocations,
                                int *unfilteredCorner2d) {

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > params.threads)
    return;

  float det = eigenValue2[tid] * eigenValue1[tid];

  // int cornerValue = (float)(abs(det)) * 1000;
  int cornerValue = (float)(abs(det)) / 100000;

  float thresh = params.cornerThresh / 800;
  if (eigenValue1[tid] > 0 && eigenValue2[tid] < 0 &&
      cornerValue > (thresh * (*maxCornerValue))) {

    int index = atomicAdd(unfilteredCount, 1);
    unfilteredLocations[index] = tid;

    unfilteredCorner2d[tid] = 1;
  }
}

void Hessian(HessianParams params, byte *image, float *gaussian, float *xx,
             float *yy, float *xy, float *firstOrderDerivativeX,
             float *firstOrderDerivativeY, float *eigenValue1,
             float *eigenValue2, int *maxCornerValue, int *unfilteredCount,
             int *unfilteredLocations, int *unfilteredCorner2d,
             cudaStream_t &stream) {
  int blocks = ceil(params.threads / 512.0);

  GaussianKernel<<<blocks, 512>>>(params, image, gaussian);

  cudaGetError("Gaussian");

  FirstOrderDerivativeKernel<<<blocks, 512, 0, stream>>>(
      params, gaussian, xx, xy, yy, firstOrderDerivativeX,
      firstOrderDerivativeY);

  cudaGetError("First Order");
  HessianKernel<<<blocks, 512, 0, stream>>>(
      params, xx, xy, yy, firstOrderDerivativeX, firstOrderDerivativeY,
      eigenValue1, eigenValue2, maxCornerValue);
  cudaGetError("Hessian");

  GetCornerKernel<<<blocks, 512, 0, stream>>>(
      params, eigenValue1, eigenValue2, maxCornerValue, unfilteredCount,
      unfilteredLocations, unfilteredCorner2d);

  cudaGetError("Get Corner");
}