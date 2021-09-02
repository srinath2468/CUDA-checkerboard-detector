#ifndef _CHECKERBOARD_HPP_
#define _CHECKERBOARD_HPP_

#include "checkerboard/corners.h"
#include "checkerboard/lines.h"
#include "common.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace jcv {

class Checkerboard {

public:
  bool Initialise(int2_t resolution, int2_t boardSize);
  void Detect(cv::Mat image);
  void Release();

private:
  // Streams
  cudaStream_t stream;

  // Variables
  std::vector<std::vector<int>> rgbPoints;
  Corners rgbCorners;
  unsigned char *d_rgbImage;
  Lines lines;

  float2_t startRegion;
  float2_t endRegion;
  int2_t boardLengths;
  int2_t _resolution;
  int side;

  // Status
  bool releasing = false;
  bool detectingColour = false;
  bool startColourDetection = false;
  bool stoppedColourDetection = false;

  bool detectingIR = false;
  bool startIRDetection = false;
  bool stoppedIRDetection = false;
};
}

#endif