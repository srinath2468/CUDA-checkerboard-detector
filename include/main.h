#ifndef _MAIN_H_
#define _MAIN_H_

#include "common.h"
#include "corners.h"
#include "lines.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Functions
void DetectColourBoard();
void DetectIRBoard();

// Threads
std::thread t_detectColour;
std::thread t_detectIR;

// Streams
cudaStream_t rgbStream;
cudaStream_t irStream;
std::vector<cudaStream_t> rgbStreamVec(4);
std::vector<cudaStream_t> irStreamVec(4);

// Variables
std::vector<std::vector<int>> rgbPoints;
std::vector<std::vector<int>> irPoints;
Corners rgbCorners;
Corners irCorners;
byte *d_rgbImage;
byte *d_irImage;
std::vector<Lines> rgbLineObjVec(4);
std::vector<Lines> irLineObjVec(4);

float2_t startRegion;
float2_t endRegion;
int2_t boardLengths;
int side;

// Status
bool releasing = false;
bool detectingColour = false;
bool startColourDetection = false;
bool stoppedColourDetection = false;

bool detectingIR = false;
bool startIRDetection = false;
bool stoppedIRDetection = false;

#endif