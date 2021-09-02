#include "checkerboard/cuda/refineLines.h"

__global__ void RefineLinesKernel(RefineLinesParams params, int *lines,
                                  int *linesPointCount, int *linePoints,
                                  int *discardedLinePoints,
                                  int *discardedLinePointCount) {

  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  int threads = params.count;
  if (threadId >= threads)
    return;

  // point
  // int px, py, px2, py2;

  // px = lines[threadId * 2] % params.resolution.x;
  // py = lines[threadId * 2] / params.resolution.x;

  // px2 = lines[threadId * 2 + 1] % params.resolution.x;
  // py2 = lines[threadId * 2 + 1] / params.resolution.x;

  int point1 = lines[threadId * 2];
  int point2 = lines[threadId * 2 + 1];

  for (int i = 0; i < params.count; i++) {

    if (threadId == i || (linesPointCount[threadId] >= linesPointCount[i]))
      continue;

    int linePoint1 = lines[i * 2];
    int linePoint2 = lines[i * 2 + 1];

    bool pxFound = false;
    bool px2Found = false;

    for (int j = 0; j < linesPointCount[i]; j++) {
      if (point1 == linePoints[i * 30 + j])
        pxFound = true;
      if (point2 == linePoints[i * 30 + j])
        px2Found = true;
    }
  }

  int index = atomicAdd(discardedLinePointCount, 1);
  discardedLinePoints[index] = threadId;
}

void RefineLines(RefineLinesParams params, int *lines, int *linesPointCount,
                 int *linePoints, int *discardedLinePoints,
                 int *discardedLinePointCount) {

  int threads = params.count;
  int blocks = ceil(threads / 128.0f);

  RefineLinesKernel<<<blocks, 128.0f>>>(params, lines, linesPointCount,
                                        linePoints, discardedLinePoints,
                                        discardedLinePointCount);
}