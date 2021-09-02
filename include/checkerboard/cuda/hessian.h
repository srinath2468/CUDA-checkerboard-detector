#ifndef _HESSIAN_H_
#define _HESSIAN_H_

#include "checkerboard/cuda/cuda_error_check.h"
#include "common.h"

// 7x7 sobel kernel for y
// 3   2   1   0   -1  -2  -3
// 4   3   2   0   -2  -3  -4
// 5   4   3   0   -3  -4  -5
// 6   5   4   0   -4  -5  -6
// 5   4   3   0   -3  -4  -5
// 4   3   2   0   -2  -3  -4
// 3   2   1   0   -1  -2  -3

// 3  4  5  6  5  4  3
// 2  3  4  5  4  3  2
// 1  2  3  4  3  2  1
// 0  0  0  0  0  0  0
// -1 -2 -3 -4 -3 -2 -1
// -2 -3 -4 -5 -4 -3 -2
// -3 -4 -5 -6 -5 -4 -3

__device__ const int sobel7x7[49][2] = {

    {3, 3},  // 0
    {2, 4},  // 1
    {1, 5},  // 2
    {0, 6},  // 3
    {-1, 5}, // 4
    {-2, 4}, // 5
    {-3, 3}, // 6

    {4, 2},  // 0
    {3, 3},  // 1
    {2, 4},  // 2
    {0, 5},  // 3
    {-2, 4}, // 4
    {-3, 3}, // 5
    {-4, 2}, // 6

    {5, 1},  // 0
    {4, 2},  // 1
    {3, 3},  // 2
    {0, 4},  // 3
    {-3, 3}, // 4
    {-4, 2}, // 5
    {-5, 1}, // 6

    {6, 0},  // 0
    {5, 0},  // 1
    {4, 0},  // 2
    {0, 0},  // 3
    {-4, 0}, // 4
    {-5, 0}, // 5
    {-6, 0}, // 6

    {5, -1},  // 0
    {4, -2},  // 1
    {3, -3},  // 2
    {0, -4},  // 3
    {-3, -3}, // 4
    {-4, -2}, // 5
    {-5, -1}, // 6

    {4, -2},  // 0
    {3, -3},  // 1
    {2, -4},  // 2
    {0, -5},  // 3
    {-2, -4}, // 4
    {-3, -3}, // 5
    {-4, -2}, // 6

    {3, -3},  // 0
    {2, -4},  // 1
    {1, -5},  // 2
    {0, -6},  // 3
    {-1, -5}, // 4
    {-2, -4}, // 5
    {-3, -3}  // 6
};

__device__ const float gaussianKernel[9] = {
    0.024879f, // 0
    0.107973f, // 1
    0.024879f, // 2
    0.107973f, // 3
    0.468592f, // 4
    0.107973f, // 5
    0.024879f, // 6
    0.107973f, // 7
    0.024879f  // 8

};

__device__ const int sobel[9][2] = {

    {-1, -1}, // 0
    {0, -2},  // 1
    {1, -1},  // 2
    {-2, 0},  // 3
    {0, 0},   // 4
    {2, 0},   // 5
    {-1, 1},  // 6
    {0, 2},   // 7
    {1, 1}    // 8
};

struct HessianParams {

  int2_t resolution;
  int threads;
  float cornerThresh;
};

void Hessian(HessianParams params, byte *image, float *gaussian, float *xx,
             float *yy, float *xy, float *firstOrderDerivativeX,
             float *firstOrderDerivativeY, float *eigenValue1,
             float *eigenValue2, int *maxCornerValue, int *unfilteredCount,
             int *unfilteredLocations, int *unfilteredCorner2d,
             cudaStream_t &stream);

#endif