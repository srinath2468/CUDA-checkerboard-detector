#include "cuda/getLines.h"

__global__ void GetLinesKernel(GetLinesParams params, int *corners,
                               int *linesCount, int *lines, int *linePoints) {

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

  bool thisIsKernel = false;
  if (corner % 1920 > (1224 * scaleX) && corner % 1920 < (1244 * scaleX) &&
      corner / 1920 > (169 * scaleY) && corner / 1920 < (192 * scaleY) &&
      compareCorner % 1920 > (1223 * scaleX) &&
      compareCorner % 1920 < (1242 * scaleX) &&
      compareCorner / 1920 > (639 * scaleY) &&
      compareCorner / 1920 < (657 * scaleY))
    thisIsKernel = true;

  // Line segment u and v
  int ux, vx, uy, vy;

  ux = corner % params.resolution.x;
  uy = corner / params.resolution.x;
  vx = compareCorner % params.resolution.x;
  vy = compareCorner / params.resolution.x;

  if (params.rows < params.cols && uy > vy)
    return;

  if (params.cols < params.rows && ux > vx)
    return;

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

      if (pointCount > (params.rows - 2))
        return;
    }
  }

  bool lineFail = false;
  if (pointCount == (params.rows - 2)) {

    float distanceArr[maxSize] = {};
    int xArr[maxSize] = {};
    int yArr[maxSize] = {};

    int arrangePointCount = 0;
    for (int i = 0; i < params.rows - 2; i++) {

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
    for (int i = 0; i < params.rows - 1; i++) {
      for (int j = i + 1; j < params.rows - 1; j++) {

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

    for (int i = 0; i < params.rows - 2; i++) {
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

    for (int i = 0; i < params.rows - 1; i++) {
      if (xArr[i] == 0 || yArr[i] == 0) {
        printf("something wrong in kernel , index = %d \n", i);
      }
    }

    if (!lineFail) {
      int index = atomicAdd(linesCount, 1);

      lines[index * 2] = corner;
      lines[index * 2 + 1] = compareCorner;

      linePoints[index * 30] = uy * params.resolution.x + ux;
      for (int j = 1; j < params.rows; j++) {
        linePoints[index * 30 + j] =
            yArr[j - 1] * params.resolution.x + xArr[j - 1];
      }
    }
  }
}

void GetLines(GetLinesParams params, int *corners, int *linesCount, int *lines,
              int *linePoints) {
  int threads = params.cornerCount * (params.cornerCount - 1);
  int blocks = ceil(threads / 128.0f);

  GetLinesKernel<<<blocks, 128.0f>>>(params, corners, linesCount, lines,
                                     linePoints);
}