#include "cuda/getLines.h"

__global__ void GetLinesKernel(GetLinesParams params, int *corners,
                               int *linesCount, int *lines, int *linePoints,
                               int *linePointCount) {

  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  int threads = params.cornerCount * (params.cornerCount - 1);
  if (threadId > threads)
    return;

  int mod = params.cornerCount - 1;
  int cornerId = threadId / mod;
  int compareId = threadId % mod;

  float scaleX = 1920.0f / 1280.0f;
  float scaleY = 1080.0f / 720.0f;

  int corner = corners[cornerId];

  int compareIdReal;
  if (compareId < cornerId)
    compareIdReal = compareId;
  else
    compareIdReal = compareId + 1;

  int compareCorner = corners[compareIdReal];

  // Line segment u and v
  int ux, vx, uy, vy;

  ux = corner % params.resolution.x;
  uy = corner / params.resolution.x;
  vx = compareCorner % params.resolution.x;
  vy = compareCorner / params.resolution.x;

  int maxSide;
  if (params.rows > params.cols)
    maxSide = params.rows;
  else
    maxSide = params.cols;

  // if (params.rows < params.cols && uy > vy)
  //   return;

  // if (params.cols < params.rows && ux > vx)
  //   return;

  float denominator = powf(vx - ux, 2) + powf(vy - uy, 2);
  float diffX = vx - ux;
  float diffY = vy - uy;
  float rootDenominator = sqrtf(denominator);

  const int maxSize = 24;

  // point
  int px, py;

  int points[maxSize] = {};

  int pxDiffFromLine;
  if (params.resolution.x < 1000)
    pxDiffFromLine = 4;
  else
    pxDiffFromLine = 5;
  int pointCount = 0;
  for (int i = 0; i < params.cornerCount; i++) {
    if (corner == corners[i] || compareCorner == corners[i])
      continue;

    px = corners[i] % params.resolution.x;
    py = corners[i] / params.resolution.x;
    float t = -1 * ((diffX * (ux - px)) + (diffY * (uy - py))) / denominator;

    if (t < 0 || t > 1)
      continue;

    float distance =
        fabsf((diffX * (uy - py)) - (diffY * (ux - px))) / rootDenominator;

    if (distance < pxDiffFromLine) {

      points[pointCount] = corners[i];

      if (corners[i] == 0) {
        printf("wrong corner accessed %d \n", i);
      }
      ++pointCount;

      if (pointCount > (maxSide - 2))
        return;
    }
  }

  bool lineFail = false;
  if (pointCount == (params.rows - 2) || pointCount == (params.cols - 2)) {

    int sizeOfLine = pointCount;

    float distanceArr[maxSize] = {};
    int xArr[maxSize] = {};
    int yArr[maxSize] = {};

    int arrangePointCount = 0;
    for (int i = 0; i < sizeOfLine; i++) {

      int x = points[i] % params.resolution.x;
      int y = points[i] / params.resolution.x;

      float dist = powf(ux - x, 2) + powf(uy - y, 2);
      distanceArr[arrangePointCount] = dist;
      xArr[arrangePointCount] = x;
      yArr[arrangePointCount] = y;

      ++arrangePointCount;
    }
    distanceArr[arrangePointCount] = powf(ux - vx, 2) + powf(uy - vy, 2);
    xArr[arrangePointCount] = vx;
    yArr[arrangePointCount] = vy;

    // sorting distance array
    for (int i = 0; i < sizeOfLine + 1; i++) {
      for (int j = i + 1; j < sizeOfLine + 1; j++) {

        if (distanceArr[i] == 0 || distanceArr[j] == 0)
          continue;

        if (distanceArr[i] > distanceArr[j]) {
          float a = distanceArr[i];
          distanceArr[i] = distanceArr[j];
          distanceArr[j] = a;

          int ax = xArr[i];
          xArr[i] = xArr[j];
          xArr[j] = ax;

          int ay = yArr[i];
          yArr[i] = yArr[j];
          yArr[j] = ay;
        }
      }
    }

    for (int i = 0; i < sizeOfLine; i++) {
      float dist1;
      if (i == 0)
        dist1 = powf(xArr[i] - ux, 2) + powf(yArr[i] - uy, 2);
      else
        dist1 = powf(xArr[i] - xArr[i - 1], 2) + powf(yArr[i] - yArr[i - 1], 2);

      float dist2 =
          powf(xArr[i + 1] - xArr[i], 2) + powf(yArr[i + 1] - yArr[i], 2);

      float percentageDiff = fabsf(dist1 - dist2) / dist2;

      if (100 * percentageDiff > 60) {
        lineFail = true;
        break;
      }
    }

    for (int i = 0; i < sizeOfLine + 1; i++) {
      if (xArr[i] == 0 || yArr[i] == 0) {
        printf("something wrong in kernel , index = %d \n", i);
      }
    }

    if (!lineFail) {
      int index = atomicAdd(linesCount, 1);

      bool horizontal = false;
      bool vertical = false;

      int point1[2] = {};
      int point2[2] = {};

      // // generating line equation
      // float yGradient = vy - uy;
      // if (yGradient == 0)
      //   horizontal = true;
      // float xGradient = vx - ux;
      // if (xGradient == 0)
      //   vertical = true;

      // if (!horizontal && !vertical) {

      //   // y=mx+b
      //   float m = yGradient / xGradient;
      //   int b = uy - (m * ux);

      //   // left point
      //   if (b == 0 || b < params.resolution.y && b > 0) {
      //     point1[0] = 0;
      //     point1[1] = b;
      //   } else {

      //     // up or down
      //     if (b >= params.resolution.y) { // down
      //       point1[0] = (params.resolution.y - b) / m;
      //       point1[1] = params.resolution.y;
      //     } else { // up
      //       point1[0] = -b / m;
      //       point1[1] = 0;
      //     }
      //   }

      //   // right point
      //   int b2 = params.resolution.x * m + b;
      //   if (b2 == 0 || b2 < params.resolution.y && b2 > 0) {
      //     point2[0] = params.resolution.x;
      //     point2[1] = b2;

      //   } else {
      //     // up or down
      //     if (b2 >= params.resolution.y) { // down
      //       point2[0] = (params.resolution.y - b) / m;
      //       point2[1] = params.resolution.y;
      //     } else { // up
      //       point2[0] = -b / m;
      //       point2[1] = 0;
      //     }
      //   }

      // } else {
      //   // x = n line
      //   if (vertical) {
      //     point1[0] = ux;
      //     point1[1] = 0;
      //     point2[0] = ux;
      //     point2[1] = params.resolution.y;
      //   } else if (horizontal) {
      //     point1[0] = 0;
      //     point1[1] = uy;
      //     point2[0] = params.resolution.x;
      //     point2[1] = ux;
      //   }
      // }

      lines[index * 2] = corner;
      lines[index * 2 + 1] = compareCorner;
      linePointCount[index] = pointCount + 2;
      linePoints[index * 30] = uy * params.resolution.x + ux;
      for (int j = 1; j < (sizeOfLine + 2); j++) {
        linePoints[index * 30 + j] =
            yArr[j - 1] * params.resolution.x + xArr[j - 1];
      }
    }
  }
}

void GetLines(GetLinesParams params, int *corners, int *linesCount, int *lines,
              int *linePoints, int *linePointCount) {
  int threads = params.cornerCount * (params.cornerCount - 1);
  int blocks = ceil(threads / 128.0f);

  GetLinesKernel<<<blocks, 128.0f>>>(params, corners, linesCount, lines,
                                     linePoints, linePointCount);
}