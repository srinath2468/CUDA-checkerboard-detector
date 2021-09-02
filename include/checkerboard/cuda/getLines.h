#ifndef _GETLINES_H_
#define _GETLINES_H_

#include "checkerboard/cuda/cuda_error_check.h"
#include "common.h"

struct GetLinesParams {
  int2_t resolution;
  int cornerCount;
  int rows;
  int cols;
  int pixelDifference;
};

void GetLines(GetLinesParams params, int *corners, int *linesCount, int *lines,
              int *linePoints, int *linePointCount, cudaStream_t &stream);

#endif