#include "checkerboard/cuda/cornersCuda.h"

__global__ void FilterCornerKernel(CornerParams params, int *unfilteredCorners,
                                   int *filteredCorners,
                                   int *filteredCornerCount) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > params.threads)
    return;

  if (unfilteredCorners[tid] == 0)
    return;

  // X and Y coordinates
  int Iy = floorf(tid / params.resolution.x); // Y index of Matrix
  int Ix = tid % params.resolution.x;         // X index of Matrix

  int count = 0;
  int x = 0;
  int y = 0;
  for (int i = -10; i < 10; i++) {
    for (int j = -10; j < 10; j++) {

      int neighbour = (Iy + j) * params.resolution.x + (Ix + i);

      if (Iy + j >= params.resolution.y || Iy + j <= 0 || Ix + i <= 0 ||
          Ix + i >= params.resolution.x)
        continue;

      if (unfilteredCorners[neighbour] != 0) {
        count++;
        x += (Ix + i);
        y += (Iy + j);
      }
    }
  }

  int newX = x / count;
  int newY = y / count;
  int newTid = newY * params.resolution.x + newX;

  if (newTid < params.resolution.x * params.resolution.y) {
    int index = atomicAdd(filteredCornerCount, 1);
    filteredCorners[index] = newTid;
  }
}

void FilterCorners(CornerParams params, int *unfilteredCorners,
                   int *filteredCorners, int *filteredCornerCount,
                   cudaStream_t &stream) {

  int blocks = ceil(params.threads / 256);

  FilterCornerKernel<<<blocks, 256, 0, stream>>>(
      params, unfilteredCorners, filteredCorners, filteredCornerCount);
}

__global__ void RemoveFakeCornerKernel(CornerParams params, int *fakeCorners,
                                       int *realCorners, int *realCornerCount,
                                       float *gaussian) {
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId > params.fakeCornerCount - 1)
    return;

  int tid = fakeCorners[threadId];

  // X and Y coordinates
  int Iy = floorf(tid / params.resolution.x); // Y index of Matrix
  int Ix = tid % params.resolution.x;         // X index of Matrix

  int numberOfChanges = 0;

  float currentIntensity = 0, nextItensity = 0;

  float maxIntensity = 0;
  float minIntensity = 100000;
  for (int i = -6; i < 7; i++) {
    for (int j = -6; j < 7; j++) {
      int neighbourId = (Iy + j) * params.resolution.x + (Ix + i);

      if (Iy + j < 2 || Iy + j > (params.resolution.y - 2) || Ix + i < 2 ||
          Ix + i > (params.resolution.x - 2))
        continue;

      if (gaussian[neighbourId] < minIntensity)
        minIntensity = gaussian[neighbourId];

      if (gaussian[neighbourId] > maxIntensity)
        maxIntensity = gaussian[neighbourId];
    }
  }

  int range = maxIntensity - minIntensity;

  // centrosymmetric test
  for (int i = 0; i < 32; i++) {
    int x = centroOffsets[i][0];
    int y = centroOffsets[i][1];

    int xNext = centroOffsets[i + 1][0];
    int yNext = centroOffsets[i + 1][1];

    int neighbourId = (Iy + y) * params.resolution.x + (Ix + x);
    int nextNeighbourId = (Iy + yNext) * params.resolution.x + (Ix + xNext);

    if ((Iy + y) < 2 || (Iy + y) > (params.resolution.y - 2) || (Ix + x) < 2 ||
        (Ix + x) > (params.resolution.x - 2))
      continue;

    if (tid == 1629666) {
      printf(" x %d and y %d AND I %d \n", x, y, i);
      printf(" neighbourhood id %d  and next %d \n", neighbourId,
             nextNeighbourId);
    }

    if (gaussian[neighbourId] > (minIntensity + (0.5f * range)))
      currentIntensity = 255;
    else
      currentIntensity = 0;

    if (gaussian[nextNeighbourId] > (minIntensity + (0.5f * range)))
      nextItensity = 255;
    else
      nextItensity = 0;

    if (tid == 1629666) {
      printf(" current intensity %f and next %f and i %d \n ", currentIntensity,
             nextItensity, i);
    }

    if (currentIntensity != nextItensity)
      ++numberOfChanges;
  }

  bool fake = false;

  if (numberOfChanges != 4) {
    fake = true;
  }

  if (fake)
    return;

  int index = atomicAdd(realCornerCount, 1);
  realCorners[tid] = 1;
}

void RemoveFakeCorners(CornerParams params, int *fakeCorners, int *realCorners,
                       int *realCornerCount, float *gaussian,
                       cudaStream_t &stream) {

  int threads = 0;
  if (params.fakeCornerCount >= 10000)
    threads = 256;
  else
    threads = 128;

  int blocks = ceil(params.fakeCornerCount / threads);

  RemoveFakeCornerKernel<<<blocks, threads, 0, stream>>>(
      params, fakeCorners, realCorners, realCornerCount, gaussian);
}

__global__ void ErodedCornerKernel(CornerParams params, int *realCorners,
                                   int *erodedCorners, int *erodedCornerCount) {

  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId > params.realCornerCount - 1)
    return;

  int tid = realCorners[threadId];

  // X and Y coordinates
  int y = floorf(tid / params.resolution.x); // Y index of Matrix
  int x = tid % params.resolution.x;         // X index of Matrix

  float thresh = 10 * 10;
  for (int i = 0; i < params.realCornerCount; i++) {
    if (i == threadId)
      continue;

    int nx = realCorners[i] % params.resolution.x;
    int ny = realCorners[i] / params.resolution.x;

    if ((powf(nx - x, 2) + powf(ny - y, 2)) < thresh)
      return;
  }

  int index = atomicAdd(erodedCornerCount, 1);
  erodedCorners[index] = tid;
}

void ErodedCorners(CornerParams params, int *realCorners, int *erodedCorners,
                   int *erodedCornerCount, cudaStream_t &stream) {

  int blocks = ceil(params.realCornerCount / 32.0);

  ErodedCornerKernel<<<blocks, 32, 0, stream>>>(
      params, realCorners, erodedCorners, erodedCornerCount);
}
