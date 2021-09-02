#ifndef _DETECTCHECKER_H_
#define _DETECTCHECKER_H_

#include "common.h"
#include "cuda/corners.h"
#include "cuda/cuda_error_check.h"
#include "cuda/getLines.h"
#include "cuda/hessian.h"
#include "cuda/refineLines.h"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <vector>

struct lineGroups {
  std::vector<int> indices;
  std::vector<int> lineLeft;
  std::vector<int> lineRight;
  float min = 360;
  float max = 0;
  int count = 0;
  float angle = 0;
  float mean = 0;
  float totalAngle = 0;
};

class CheckerBoard {

public:
  void Initialise(int2_t resolutionImage);

  int none = 0;
  int left = 1;
  int right = 2;
  int top = 3;
  int bot = 4;
  int leftAndRight = 5;
  int four = 6;

  bool Detect(byte *image, int rows, int cols,
              std::vector<std::vector<int>> &imagePoints, float2_t startRatio,
              float2_t endRatio, int side, int *&debugPoints,
              int &debugPointsSize, std::vector<lineGroups> &linesVec);

private:
  int maxPoints;

  bool GetCheckerBoard(byte *image, int rows, int cols,
                       std::vector<int> &imagePoints, float2_t startRatio,
                       float2_t endRatio, int side, int *&debugPoints,
                       int &debugPointsSize, std::vector<lineGroups> &linesVec);

  int2_t resolution;

  byte *imagePtr;
  byte *d_image;
  float *d_gaussian;
  byte *d_byteGaussian;
  byte *p_gaussian;

  float *d_firstOrderDerivativeX;
  float *d_firstOrderDerivativeY;
  float *p_firstOrderDerivativeX;

  float *d_eigenValue1;
  float *d_eigenValue2;

  int *d_maxCornerVal;

  int *d_unfilteredCornerLocations;

  int *d_filteredCornerLocations;
  int *d_unfilteredCornerCount;
  int *p_unfilteredCornerCount;
  int *p_unfilteredCornerLocations;
  int *p_filteredCornerLocations;

  int *p_fakeCornerLocations;
  int *d_fakeCornerLocations;
  int *p_realCornerLocations;
  int *d_realCornerLocations;
  int *d_realCornerCount;
  int *p_realCornerCount;

  int *d_discaredAngleCount;
  int *d_discaredDistanceCount;
  int *d_discaredCentroSymCount;
  int *p_discaredAngleCount;
  int *p_discaredDistanceCount;
  int *p_discaredCentroSymCount;
  int *d_discaredAngleArr;
  int *d_discaredDistanceArr;
  int *d_discaredCentroSymArr;
  int *p_discaredAngleArr;
  int *p_discaredDistanceArr;
  int *p_discaredCentroSymArr;

  float *d_xx;
  float *d_yy;
  float *d_xy;

  int *d_corners;
  int *p_corners;

  int frameCount = 0;

  HessianParams hessianParams;

  CornerParams cornerParams;

  RefineLinesParams refineLinesParams;

  rs2::sensor colourSensor;

  int brightness;
  int contrast;
  int exposure;
  int gain;

  GetLinesParams getLinesParams;
  int *d_linesCount;
  int *p_linesCount;
  int *d_lines;
  int *p_lines;

  int *d_erodedCornerCount;
  int *d_erodedCornerLocations;
  int *p_erodedCornerCount;
  int *p_erodedCornerLocations;

  int *d_linePoints;
  int *p_linePoints;
  int *d_linePointCount;
  int *p_linePointCount;
  int *d_discardedLinePoints;
  int *p_discardedLinePoints;
  int *d_discardedLinePointCount;
  int *p_discardedLinePointCount;

  cv::Mat image;
  cv::Mat imageUntouched;
  std::vector<int> imagePoints;
  int saveCount;

  float2_t start, end;

  // Printing images
  byte *gaussArr;

  std::vector<lineGroups> lineGroupVec;
  std::vector<lineGroups> lineGroupVecMerged;
};

#endif