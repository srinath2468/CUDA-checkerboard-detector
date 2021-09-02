#ifndef _GET_CORNERS_H_
#define _GET_CORNERS_H_

#include "checkerboard/cuda/cornersCuda.h"
#include "checkerboard/cuda/hessian.h"
#include "common.h"
#include <opencv2/opencv.hpp>

// std::vector<int> GetCornersRGB(byte *image, cudaStream_t &stream) {}

namespace jcv {

class Corners {

public:
  bool InitialiseCorners(int2_t resolution);
  bool Detect(byte *&image, cudaStream_t &stream, std::vector<int> &corners);
  bool Release();

private:
  // Params
  HessianParams hessianParams;
  CornerParams cornerParams;
  // variables
  int maxPoints;

  float *d_gaussian;              // Device
  float *d_xx;                    // Device
  float *d_xy;                    // Device
  float *d_yy;                    // Device
  float *d_eigenValue1;           // Device
  float *d_eigenValue2;           // Device
  float *d_firstOrderDerivativeX; // Device
  float *d_firstOrderDerivativeY; // Device

  int *d_unfilteredCornerCount;     // Device
  int *d_maxCornerVal;              // Device
  int *d_unfilteredCorners;         // Device
  int *d_unfilteredCornerLocations; // Device
  int *d_filteredCornerLocations;   // Device
  int *d_filteredCornerCount;       // Device
  int *p_filteredCornerCount;       // Pinned
  int *filteredCornerLocations;     // Host
  int *p_unfilteredCornerCount;     // Pinned
  int *p_unfilteredCornerLocations; // Pinned
  int *d_unfilteredCorner2d;        // Device

  int *d_realCornerLocations; // Device
  int *p_realCornerLocations; // Pinned
  int *d_realCornerCount;     // Device
  int *p_realCornerCount;     // Pinned
  int *d_fakeCornerLocations; // Device
  int *fakeCornerLocations;   // Host

  int *d_erodedCornerLocations; // Device
  int *d_erodedCornerCount;     // Device
  int *p_erodedCornerCount;     // Pinned
  int *cornerLocations;         // Host

  // REMOVE LATER
  float *p_gaussian; // Pinned
  byte *p_bGaussian; // Pinned
  int iter = 0;
};
}
#endif