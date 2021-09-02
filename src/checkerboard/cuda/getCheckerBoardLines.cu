#include "checkerboard/cuda/getLines.h"

__global__ void GetLinesKernel(GetLinesParams params, int *corners,
                               int *linesCount, int *lines, int *linePoints,
                               int *linePointCount) {

  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  int threads = params.cornerCount * (params.cornerCount - 1);
  if (threadId > threads)
    return;

  const int maxSize = 40;

  int mod = params.cornerCount - 1;
  int cornerId = threadId / mod;
  int compareId = threadId % mod;

  int corner = corners[cornerId];

  int compareIdReal;
  if (compareId < cornerId)
    compareIdReal = compareId;
  else
    compareIdReal = compareId + 1;

  int compareCorner = corners[compareIdReal];

  int maxSide, minSide;
  if (params.rows > params.cols) {
    maxSide = params.rows;
    minSide = params.cols;
  } else {
    maxSide = params.cols;
    minSide = params.rows;
  }

  // Line segment u and v
  int ux, vx, uy, vy;

  ux = corner % params.resolution.x;
  uy = corner / params.resolution.x;
  vx = compareCorner % params.resolution.x;
  vy = compareCorner / params.resolution.x;

  float denominator = powf(vx - ux, 2) + powf(vy - uy, 2);
  float diffX = vx - ux;
  float diffY = vy - uy;
  float rootDenominator = sqrtf(denominator);

  int points1[maxSize] = {};
  bool lineFail = false;

  // point
  int px, py;
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

      points1[pointCount] = corners[i];

      ++pointCount;
    }
  }

  if (pointCount < (minSide - 2))
    return;

  int sizeOfLine1 = pointCount;

  float distanceArr1[maxSize] = {};
  int xArr1[maxSize] = {};
  int yArr1[maxSize] = {};

  int arrangePointCount1 = 0;
  for (int i = 0; i < sizeOfLine1; i++) {

    int x = points1[i] % params.resolution.x;
    int y = points1[i] / params.resolution.x;

    float dist = powf(ux - x, 2) + powf(uy - y, 2);

    distanceArr1[arrangePointCount1] = dist;
    xArr1[arrangePointCount1] = x;
    yArr1[arrangePointCount1] = y;

    ++arrangePointCount1;
  }
  distanceArr1[arrangePointCount1] = powf(ux - vx, 2) + powf(uy - vy, 2);
  xArr1[arrangePointCount1] = vx;
  yArr1[arrangePointCount1] = vy;

  // sorting distance array
  for (int i = 0; i < arrangePointCount1; i++) {
    for (int j = i + 1; j < arrangePointCount1; j++) {

      if (distanceArr1[i] == 0 || distanceArr1[j] == 0)
        continue;

      if (distanceArr1[i] > distanceArr1[j]) {
        float a = distanceArr1[i];
        distanceArr1[i] = distanceArr1[j];
        distanceArr1[j] = a;

        int ax = xArr1[i];
        xArr1[i] = xArr1[j];
        xArr1[j] = ax;

        int ay = yArr1[i];
        yArr1[i] = yArr1[j];
        yArr1[j] = ay;
      }
    }
  }

  for (int i = 0; i < pointCount; i++) {
    float dist1;
    if (i == 0)
      dist1 = powf(xArr1[i] - ux, 2) + powf(yArr1[i] - uy, 2);
    else
      dist1 =
          powf(xArr1[i] - xArr1[i - 1], 2) + powf(yArr1[i] - yArr1[i - 1], 2);

    float dist2 =
        powf(xArr1[i + 1] - xArr1[i], 2) + powf(yArr1[i + 1] - yArr1[i], 2);

    float percentageDiff = fabsf(dist1 - dist2) / dist2;

    if (100 * percentageDiff > 60) {

      lineFail = true;
      break;
    }
  }

  if (lineFail)
    return;

  // generating line equation

  bool horizontal = false;
  bool vertical = false;

  int point1[2] = {};
  int point2[2] = {};

  float yGradient = vy - uy;
  if (yGradient == 0)
    horizontal = true;
  float xGradient = vx - ux;
  if (xGradient == 0)
    vertical = true;

  if (!horizontal && !vertical) {

    // y=mx+b
    float m = yGradient / xGradient;
    int b = uy - (m * ux);

    // left point
    if (b == 0 || b < params.resolution.y - 1 && b > 0) {
      point1[0] = 0;
      point1[1] = b;
    } else {

      // up or down
      if (b >= params.resolution.y) { // down
        point1[0] = (params.resolution.y - b) / m;
        point1[1] = params.resolution.y - 1;
      } else { // up
        point1[0] = -b / m;
        point1[1] = 0;
      }
    }

    // right point
    int b2 = params.resolution.x * m + b;
    if (b2 == 0 || b2 < params.resolution.y - 1 && b2 > 0) {
      point2[0] = params.resolution.x - 1;
      point2[1] = b2;

    } else {
      // up or down
      if (b2 >= params.resolution.y) { // down
        point2[0] = (params.resolution.y - b) / m;
        point2[1] = params.resolution.y - 1;
      } else { // up
        point2[0] = -b / m;
        point2[1] = 0;
      }
    }

  } else {
    // x = n line
    if (vertical) {
      point1[0] = ux;
      point1[1] = 0;
      point2[0] = ux;
      point2[1] = params.resolution.y - 1;
    } else if (horizontal) {
      point1[0] = 0;
      point1[1] = uy;
      point2[0] = params.resolution.x - 1;
      point2[1] = uy;
    }
  }

  pxDiffFromLine = params.pixelDifference;

  denominator = powf(point2[0] - point1[0], 2) + powf(point2[1] - point1[1], 2);
  diffX = point2[0] - point1[0];
  diffY = point2[1] - point1[1];
  rootDenominator = sqrtf(denominator);

  int points[maxSize] = {};

  pointCount = 0;
  for (int i = 0; i < params.cornerCount; i++) {

    px = corners[i] % params.resolution.x;
    py = corners[i] / params.resolution.x;
    float t = -1 * ((diffX * (point1[0] - px)) + (diffY * (point1[1] - py))) /
              denominator;

    if (t < 0 || t > 1)
      continue;

    float distance =
        fabsf((diffX * (point1[1] - py)) - (diffY * (point1[0] - px))) /
        rootDenominator;

    if (distance < pxDiffFromLine) {

      points[pointCount] = corners[i];

      ++pointCount;
    }

    if (pointCount == maxSize)
      return;
  }

  int sizeOfLine = pointCount;

  float distanceArr[maxSize] = {};
  int xArr[maxSize] = {};
  int yArr[maxSize] = {};

  int arrangePointCount = 0;
  for (int i = 0; i < sizeOfLine; i++) {

    int x = points[i] % params.resolution.x;
    int y = points[i] / params.resolution.x;

    float dist = powf(point1[0] - x, 2) + powf(point1[1] - y, 2);
    distanceArr[arrangePointCount] = dist;
    xArr[arrangePointCount] = x;
    yArr[arrangePointCount] = y;

    ++arrangePointCount;
  }

  // sorting distance array
  for (int i = 0; i < arrangePointCount; i++) {
    for (int j = i + 1; j < arrangePointCount; j++) {

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

  int goodPointsSize = 2;
  int startIndex = 0;

  for (int i = 1; i < sizeOfLine; i++) {
    bool badPercentageDiff = false;
    float dist1 =
        powf(xArr[i] - xArr[i - 1], 2) + powf(yArr[i] - yArr[i - 1], 2);

    float dist2 =
        powf(xArr[i + 1] - xArr[i], 2) + powf(yArr[i + 1] - yArr[i], 2);

    float percentageDiff = fabsf(dist1 - dist2) / dist2;

    if (100 * percentageDiff > 80)
      badPercentageDiff = true;

    if (badPercentageDiff && goodPointsSize < minSide) {
      if (i == 1) {
        startIndex = i;
        continue;
      }
      startIndex = i + 1;
      i++;
      goodPointsSize = 1;

    } else if (badPercentageDiff && goodPointsSize >= minSide)
      break;

    ++goodPointsSize;
  }

  if (goodPointsSize >= minSide) {

    bool secondcase = false;
    if ((ux > 1235 && ux < 1241 && uy > 229 && uy < 236) ||
        (vx > 1235 && vx < 1241 && vy > 229 && vy < 236))
      secondcase = true;

    int index = atomicAdd(linesCount, 1);

    lines[index * 2] = point1[1] * params.resolution.x + point1[0];
    lines[index * 2 + 1] = point2[1] * params.resolution.x + point2[0];
    linePointCount[index] = goodPointsSize;
    for (int j = 0; j < goodPointsSize; j++) {
      linePoints[index * maxSize + j] =
          yArr[j + startIndex] * params.resolution.x + xArr[j + startIndex];
    }
  }
}

void GetLines(GetLinesParams params, int *corners, int *linesCount, int *lines,
              int *linePoints, int *linePointCount, cudaStream_t &stream) {

  int threads = params.cornerCount * (params.cornerCount - 1);
  int blocks = ceil(threads / 128.0f);

  GetLinesKernel<<<blocks, 128.0f, 0, stream>>>(
      params, corners, linesCount, lines, linePoints, linePointCount);

  cudaGetError("GetLines");
}