#ifndef _CORNERS_H_
#define _CORNERS_H_

#include "common.h"

// __device__ const int centroOffsets[8][2] = {
//     {1, 0},   // 0
//     {1, -1},  // 1
//     {0, -1},  // 2
//     {-1, -1}, // 3
//     {-1, 0},  // 4
//     {-1, 1},  // 5
//     {0, 1},   // 6
//     {1, 1}    // 7

// };

__device__ const int centroOffsets[33][2] = {

    {-5, -3}, // 0
    {-4, -3}, // 1
    {-3, -3}, // 2
    {-2, -3}, // 3
    {-1, -3}, // 4
    {0, -3},  // 5
    {1, -3},  // 6
    {2, -3},  // 7
    {3, -3},  // 8
    {4, -3},  // 9
    {5, -3},  // 10

    {5, -2}, // 11
    {5, -1}, // 12
    {5, 0},  // 13
    {5, 1},  // 14
    {5, 2},  // 15
    {5, 3},  // 16

    {4, 3},  // 17
    {3, 3},  // 18
    {2, 3},  // 19
    {1, 3},  // 20
    {0, 3},  // 21
    {-1, 3}, // 22
    {-2, 3}, // 23
    {-3, 3}, // 24
    {-4, 3}, // 25
    {-5, 3}, // 26

    {-5, 2},  // 27
    {-5, 1},  // 28
    {-5, 0},  // 29
    {-5, -1}, // 30
    {-5, -2}, // 31
    {-5, -3}, // 32

};

// __device__ const int centroOffsets[41][2] = {
//     {-5, -5}, // 0
//     {-4, -5}, // 1
//     {-3, -5}, // 2
//     {-2, -5}, // 3
//     {-1, -5}, // 4
//     {0, -5},  // 5
//     {1, -5},  // 6
//     {2, -5},  // 7
//     {3, -5},  // 8
//     {4, -5},  // 9
//     {5, -5},  // 10

//     {5, -4}, // 11
//     {5, -3}, // 12
//     {5, -2}, // 13
//     {5, -1}, // 14
//     {5, 0},  // 15
//     {5, 1},  // 16
//     {5, 2},  // 17
//     {5, 3},  // 18
//     {5, 4},  // 19
//     {5, 5},  // 20

//     {4, 5},  // 21
//     {3, 5},  // 22
//     {2, 5},  // 23
//     {1, 5},  // 24
//     {0, 5},  // 25
//     {-1, 5}, // 26
//     {-2, 5}, // 27
//     {-3, 5}, // 28
//     {-4, 5}, // 29
//     {-5, 5}, // 30

//     {-5, 4},  // 31
//     {-5, 3},  // 32
//     {-5, 2},  // 33
//     {-5, 1},  // 34
//     {-5, 0},  // 35
//     {-5, -1}, // 36
//     {-5, -2}, // 37
//     {-5, -3}, // 38
//     {-5, -4}, // 39
//     {-5, -5}  // 40

// };

struct CornerParams {
  int2_t resolution;
  int threads;
  int count;
  int fakeCornerCount;
  int centroScale;
  int centroRatio;
  int distanceThresh;
  int angleThresh;
  int realCornerCount;
  int binaryThresh;
};

void FilterCorners(CornerParams params, int *unfilteredCorners,
                   int *filteredCorners, int *filteredCornerCount,
                   cudaStream_t &stream);

void RemoveFakeCorners(CornerParams params, int *fakeCorners, int *realCorners,
                       int *realCornerCount, float *gaussian,
                       cudaStream_t &stream);

void ErodedCorners(CornerParams params, int *realCorners, int *erodedCorners,
                   int *erodedCornerCount, cudaStream_t &stream);

#endif