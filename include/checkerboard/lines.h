#ifndef _LINES_H_
#define _LINES_H_

#include "checkerboard/cuda/getLines.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

namespace jcv {

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

class Lines {

public:
  bool Initialise(int2_t resolution);

  bool Detect(int2_t boardLengths, std::vector<int> &cornersVec,
              cudaStream_t &stream);
  std::vector<int> ReturnPoints();
  bool DetectionStatus();
  void Release();

private:
  int maxPoints;

  GetLinesParams getLinesParams;
  int2_t resolution;

  // vector of ordered points
  std::vector<int> points;

  std::thread t_detect;

  // Variables from app
  float2_t start;
  float2_t end;
  int2_t boardLengths;
  cudaStream_t stream;
  std::vector<int> cornersVec;
  int side;

  // Line group vec
  std::vector<lineGroups> lineGroupVec;
  std::vector<lineGroups> lineGroupVecMerged;

  // Lines variables
  int *d_lines;           // Device
  int *d_linesCount;      // Device
  int *d_linePoints;      // Device
  int *d_linePointCount;  // Device
  int *d_cornerLocations; // Device
  int *corners;           // Host

  int *linesCount;     // Host
  int *lines;          // Host
  int *linePoints;     // Host
  int *linePointCount; // Host

  // Status
  bool startDetection = false;
  bool releasing = false;
  bool detecting = false;
};
}

#endif