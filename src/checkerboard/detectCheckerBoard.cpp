#include "detectCheckerBoard.h"

using namespace cv;

int2_t rotatePoint(int2_t pivot, float angle, int2_t p) {

  int2_t result = int2_t();

  float s = sin(angle);
  float c = cos(angle);

  // translate point back to origin:
  p.x -= pivot.x;
  p.y -= pivot.y;

  // rotate point
  float xnew = p.x * c - p.y * s;
  float ynew = p.x * s + p.y * c;

  // translate point back:
  result.x = xnew + pivot.x;
  result.y = ynew + pivot.y;
  return result;
}

float getDist(int x1, int y1, int x2, int y2) {
  return powf(x1 - x2, 2) + powf(y1 - y2, 2);
}

void CheckerBoard::Initialise(int2_t resolutionImage) {

  resolution = resolutionImage;

  hessianParams.cornerThresh = 15;
  cornerParams.centroScale = 20;
  cornerParams.centroRatio = 50;
  cornerParams.angleThresh = 30;
  cornerParams.distanceThresh = 120;
  cornerParams.binaryThresh = 100;

  maxPoints = resolution.x * resolution.y;
  hessianParams.threads = maxPoints;
  hessianParams.resolution = resolution;

  cornerParams.threads = maxPoints;
  cornerParams.resolution = resolution;

  cudaMalloc((void **)&d_image, sizeof(byte) * maxPoints * 3);
  cudaMalloc((void **)&d_gaussian, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_byteGaussian, sizeof(byte) * maxPoints);
  cudaMallocHost((void **)&p_gaussian, sizeof(byte) * maxPoints);

  cudaMalloc((void **)&d_xx, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_yy, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_xy, sizeof(float) * maxPoints);

  cudaMalloc((void **)&d_maxCornerVal, sizeof(int));

  cudaMalloc((void **)&d_firstOrderDerivativeX, sizeof(float) * maxPoints);
  cudaMallocHost((void **)&p_firstOrderDerivativeX, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_firstOrderDerivativeY, sizeof(float) * maxPoints);

  cudaMalloc((void **)&d_unfilteredCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_filteredCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_unfilteredCornerCount, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_unfilteredCornerCount, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_unfilteredCornerLocations,
                 sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_filteredCornerLocations, sizeof(int) * maxPoints);

  cudaMallocHost((void **)&p_fakeCornerLocations, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_realCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_fakeCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_realCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_realCornerCount, sizeof(int));
  cudaMallocHost((void **)&p_realCornerCount, sizeof(int));

  cudaMalloc((void **)&d_eigenValue1, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_eigenValue2, sizeof(float) * maxPoints);

  cudaMalloc((void **)&d_corners, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_corners, sizeof(int) * maxPoints);

  cudaMalloc((void **)&d_discaredAngleCount, sizeof(int));
  cudaMalloc((void **)&d_discaredDistanceCount, sizeof(int));
  cudaMalloc((void **)&d_discaredCentroSymCount, sizeof(int));
  cudaMallocHost((void **)&p_discaredAngleCount, sizeof(int));
  cudaMallocHost((void **)&p_discaredDistanceCount, sizeof(int));
  cudaMallocHost((void **)&p_discaredCentroSymCount, sizeof(int));
  cudaMalloc((void **)&d_discaredAngleArr, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_discaredDistanceArr, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_discaredCentroSymArr, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_discaredAngleArr, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_discaredDistanceArr, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_discaredCentroSymArr, sizeof(int) * maxPoints);

  cudaMalloc((void **)&d_linesCount, sizeof(int));
  cudaMallocHost((void **)&p_linesCount, sizeof(int));
  cudaMalloc((void **)&d_lines, sizeof(int) * maxPoints * 2);
  cudaMallocHost((void **)&p_lines, sizeof(int) * maxPoints * 2);

  cudaMalloc((void **)&d_erodedCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_erodedCornerCount, sizeof(int));
  cudaMallocHost((void **)&p_erodedCornerCount, sizeof(int));

  cudaMalloc((void **)&d_linePoints, sizeof(int) * maxPoints * 40);
  cudaMallocHost((void **)&p_linePoints, sizeof(int) * maxPoints * 40);
  cudaMalloc((void **)&d_linePointCount, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_linePointCount, sizeof(int) * maxPoints);

  cudaMalloc((void **)&d_discardedLinePoints, sizeof(int) * maxPoints * 2);
  cudaMallocHost((void **)&p_discardedLinePoints, sizeof(int) * maxPoints * 2);
  cudaMalloc((void **)&d_discardedLinePointCount, sizeof(int) * maxPoints);
  cudaMallocHost((void **)&p_discardedLinePointCount, sizeof(int) * maxPoints);

  p_erodedCornerLocations = (int *)malloc(sizeof(int) * maxPoints);
}

bool CheckerBoard::Detect(byte *image, int rows, int cols,
                          std::vector<std::vector<int>> &imagePoints,
                          std::vector<lineGroups> &linesVec) {
  imagePoints.clear();
  bool detected = false;

  std::vector<int> points;
  detected = GetCheckerBoard(image, rows, cols, points, linesVec);

  imagePoints.push_back(points);

  return detected;
}

bool CheckerBoard::GetCheckerBoard(byte *image, int rows, int cols,
                                   std::vector<int> &imagePoints,
                                   std::vector<lineGroups> &linesVec) {

  float angleDiff = 20;

  start = float2_t(resolution.x, resolution.y);
  end = float2_t(resolution.x, resolution.y);

  cudaMemcpy(d_image, image, sizeof(byte) * maxPoints * 3,
             cudaMemcpyHostToDevice);

  cudaMemset(d_gaussian, 0, sizeof(float) * maxPoints);
  cudaMemset(d_xx, 0, sizeof(float) * maxPoints);
  cudaMemset(d_xy, 0, sizeof(float) * maxPoints);
  cudaMemset(d_yy, 0, sizeof(float) * maxPoints);
  cudaMemset(d_eigenValue1, 0, sizeof(float) * maxPoints);
  cudaMemset(d_eigenValue2, 0, sizeof(float) * maxPoints);
  cudaMemset(d_corners, 0, sizeof(int) * maxPoints);
  cudaMemset(d_maxCornerVal, 0, sizeof(int));
  cudaMemset(d_unfilteredCornerCount, 0, sizeof(int));
  cudaMemset(d_linePointCount, 0, sizeof(int) * maxPoints);
  cudaMemset(d_discardedLinePointCount, 0, sizeof(int));

  hessianParams.startCrop = start;
  hessianParams.endCrop = end;

  Hessian(hessianParams, d_image, d_gaussian, d_xx, d_yy, d_xy, d_corners,
          d_firstOrderDerivativeX, d_firstOrderDerivativeY, d_eigenValue1,
          d_eigenValue2, d_maxCornerVal, d_unfilteredCornerCount,
          d_unfilteredCornerLocations, d_byteGaussian);

  cudaMemcpy(p_corners, d_corners, sizeof(int) * maxPoints,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(p_gaussian, d_byteGaussian, sizeof(byte) * maxPoints,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(p_firstOrderDerivativeX, d_firstOrderDerivativeY,
             sizeof(float) * maxPoints, cudaMemcpyDeviceToHost);

  cudaMemcpy(p_unfilteredCornerCount, d_unfilteredCornerCount, sizeof(int),
             cudaMemcpyDeviceToHost);

  if (*p_unfilteredCornerCount < 10)
    return false;

  cornerParams.count = *p_unfilteredCornerCount;
  FilterCorners(cornerParams, d_unfilteredCornerLocations,
                d_filteredCornerLocations);

  cudaMemcpy(p_unfilteredCornerLocations, d_unfilteredCornerLocations,
             sizeof(int) * maxPoints, cudaMemcpyDeviceToHost);
  cudaMemcpy(p_filteredCornerLocations, d_filteredCornerLocations,
             sizeof(int) * maxPoints, cudaMemcpyDeviceToHost);

  std::vector<int> duplicateArray(p_filteredCornerLocations,
                                  p_filteredCornerLocations +
                                      cornerParams.count);

  std::sort(duplicateArray.begin(), duplicateArray.end());

  // Start traversing elements
  int n = cornerParams.count;

  if (n < 3)
    return false;

  int j = 0;
  int temp[n] = {};
  for (int i = 0; i < n - 1; i++) {

    if (duplicateArray[i] != duplicateArray[i + 1])
      p_fakeCornerLocations[j++] = duplicateArray[i];
  }
  p_fakeCornerLocations[j++] = duplicateArray[n - 1];

  cudaMemset(d_discaredAngleCount, 0, sizeof(int));
  cudaMemset(d_discaredCentroSymCount, 0, sizeof(int));
  cudaMemset(d_discaredDistanceCount, 0, sizeof(int));

  cudaMemset(d_realCornerCount, 0, sizeof(int));
  cudaMemcpy(d_fakeCornerLocations, p_fakeCornerLocations,
             sizeof(int) * maxPoints, cudaMemcpyHostToDevice);
  cornerParams.fakeCornerCount = j;

  RemoveFakeCorners(cornerParams, d_fakeCornerLocations, d_realCornerLocations,
                    d_realCornerCount, d_gaussian, d_discaredAngleCount,
                    d_discaredAngleArr, d_discaredDistanceCount,
                    d_discaredDistanceArr, d_discaredCentroSymCount,
                    d_discaredCentroSymArr);

  cudaMemcpy(p_realCornerCount, d_realCornerCount, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(p_realCornerLocations, d_realCornerLocations,
             sizeof(int) * (*p_realCornerCount), cudaMemcpyDeviceToHost);

  if (*p_realCornerCount < 10)
    return false;

  // Filter #2
  cornerParams.count = *p_realCornerCount;
  cudaMemcpy(d_unfilteredCornerLocations, d_realCornerLocations,
             sizeof(int) * maxPoints, cudaMemcpyDeviceToDevice);
  cudaMemset(d_filteredCornerLocations, 0, sizeof(int) * maxPoints);
  FilterCorners(cornerParams, d_unfilteredCornerLocations,
                d_filteredCornerLocations);

  cudaMemcpy(p_filteredCornerLocations, d_filteredCornerLocations,
             sizeof(int) * maxPoints, cudaMemcpyDeviceToHost);

  std::vector<int> duplicateArray2(p_filteredCornerLocations,
                                   p_filteredCornerLocations +
                                       cornerParams.count);

  std::sort(duplicateArray2.begin(), duplicateArray2.end());

  memset(p_fakeCornerLocations, 0, sizeof(int) * maxPoints);
  // Start traversing elements and remove duplicates
  int n2 = cornerParams.count;

  if (n2 < 3)
    return false;

  int j2 = 0;
  for (int i = 0; i < n2 - 1; i++) {

    if (duplicateArray2[i] != duplicateArray2[i + 1])
      p_fakeCornerLocations[j2++] = duplicateArray2[i];
  }
  p_fakeCornerLocations[j2++] = duplicateArray2[n2 - 1];

  cornerParams.realCornerCount = j2;
  cudaMemset(d_erodedCornerCount, 0, sizeof(int));
  cudaMemset(d_erodedCornerLocations, 0, sizeof(int) * maxPoints);
  cudaMemcpy(d_realCornerLocations, p_fakeCornerLocations,
             sizeof(int) * maxPoints, cudaMemcpyHostToDevice);
  ErodedCorners(cornerParams, d_realCornerLocations, d_erodedCornerLocations,
                d_erodedCornerCount);

  cudaMemcpy(p_erodedCornerCount, d_erodedCornerCount, sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(p_realCornerLocations, d_erodedCornerLocations,
             sizeof(int) * (*p_erodedCornerCount), cudaMemcpyDeviceToHost);

  memset(p_erodedCornerLocations, 0, sizeof(int) * maxPoints);

  int cornerCount = 0;
  for (int i = 0; i < *p_erodedCornerCount; i++) {

    int x = p_realCornerLocations[i] % resolution.x;
    int y = p_realCornerLocations[i] / resolution.x;
    if (x > start.x && x < end.x && y > start.y && y < end.y) {
      p_erodedCornerLocations[cornerCount] = p_realCornerLocations[i];
      cornerCount++;
    }
  }

  cudaMemcpy(d_erodedCornerLocations, p_erodedCornerLocations,
             sizeof(int) * cornerCount, cudaMemcpyHostToDevice);

  if (cornerCount < 10)
    return false;

  getLinesParams.cornerCount = cornerCount;
  getLinesParams.resolution = resolution;
  getLinesParams.rows = rows;
  getLinesParams.cols = cols;

  cudaMemset(d_lines, 0, sizeof(int) * maxPoints * 2);
  cudaMemset(d_linesCount, 0, sizeof(int));

  getLinesParams.pixelDifference = 6;

  GetLines(getLinesParams, d_erodedCornerLocations, d_linesCount, d_lines,
           d_linePoints, d_linePointCount);

  cudaMemcpy(p_linesCount, d_linesCount, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(p_lines, d_lines, sizeof(int) * maxPoints * 2,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(p_linePointCount, d_linePointCount, sizeof(int) * maxPoints,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(p_linePoints, d_linePoints, sizeof(int) * (*p_linesCount) * 40,
             cudaMemcpyDeviceToHost);

  if (*p_linesCount < 2)
    return false;

  int count = 0;

  float scaleX = 1920.0f / 1280.0f;
  float scaleY = 1080.0f / 720.0f;

  // Remove duplicate lines
  std::vector<int> linesPt1;
  std::vector<int> linesPt2;
  std::vector<int> lineIndices;

  int line1, line2;

  for (int i = 0; i < *p_linesCount; i++) {
    int a = p_linePoints[i * 40];
    int b = p_linePoints[i * 40 + (p_linePointCount[i] - 1)];

    bool add = true;

    if (linesPt1.size() > 0) {
      for (int j = 0; j < (int)linesPt1.size(); j++) {

        if ((a == linesPt2[j] && b == linesPt1[j]) ||
            (a == linesPt1[j] && b == linesPt2[j]))
          add = false;
      }
    }

    if (add) {

      int ux = a % resolution.x;
      int uy = a / resolution.x;
      int vx = b % resolution.x;
      int vy = b / resolution.x;

      ux = b % resolution.x;
      uy = b / resolution.x;
      vx = a % resolution.x;
      vy = a / resolution.x;

      lineIndices.push_back(i);
      linesPt1.push_back(a);
      linesPt2.push_back(b);
    }
  }

  // Get left most points in image;
  std::vector<int> leftMostPoints;
  std::vector<int> linesLeft;

  lineGroupVec.clear();

  for (int i = 0; i < linesPt1.size(); i++) {

    lineGroups member;
    std::vector<int> indiceMembers;

    int ax = linesPt1[i] % resolution.x;
    int ay = linesPt1[i] / resolution.x;
    int bx = linesPt2[i] % resolution.x;
    int by = linesPt2[i] / resolution.x;
    float angle = atan2(by - ay, bx - ax) * 180 / M_PI;
    float result = ((int)angle + 360) % 360;

    if (result >= 180)
      result = result - 180;

    if (i == 0) {

      member.angle = result;
      member.indices.push_back(lineIndices[i]);
      member.lineLeft.push_back(linesPt1[i]);
      member.lineRight.push_back(linesPt2[i]);
      member.count = 1;
      member.mean = result;
      member.totalAngle = result;
      member.min = result;
      member.max = result;
      lineGroupVec.push_back(member);

    } else {
      bool notThere = true;
      for (int j = 0; j < lineGroupVec.size(); j++) {
        if (fabs(lineGroupVec[j].angle - result) < angleDiff) {
          lineGroupVec[j].indices.push_back(lineIndices[i]);
          lineGroupVec[j].count++;
          lineGroupVec[j].lineLeft.push_back(linesPt1[i]);

          lineGroupVec[j].totalAngle += result;

          lineGroupVec[j].mean =
              lineGroupVec[j].totalAngle / lineGroupVec[j].count;

          lineGroupVec[j].lineRight.push_back(linesPt2[i]);
          if (result < lineGroupVec[j].min)
            lineGroupVec[j].min = result;
          if (result > lineGroupVec[j].max)
            lineGroupVec[j].max = result;

          notThere = false;
          break;
        }
      }

      if (notThere) {

        member.angle = result;
        member.indices.push_back(lineIndices[i]);
        member.lineLeft.push_back(linesPt1[i]);
        member.count = 1;
        member.mean = result;
        member.lineRight.push_back(linesPt2[i]);
        member.totalAngle = result;
        member.min = result;
        member.max = result;
        lineGroupVec.push_back(member);
      }
    }
  }

  std::vector<std::pair<int, int>> sameGroups;
  std::vector<float> meanAngleVec;
  // Pairing groups
  for (int i = 0; i < lineGroupVec.size(); i++) {
    int index = i;
    for (int j = 0; j < lineGroupVec.size(); j++) {
      if (i >= j)
        continue;

      if (fabs(lineGroupVec[i].min - lineGroupVec[j].min) < 10 ||
          fabs(lineGroupVec[i].min - lineGroupVec[j].max) < 10 ||
          fabs(lineGroupVec[i].max - lineGroupVec[j].min) < 10 ||
          fabs(lineGroupVec[i].max - lineGroupVec[j].max) < 10) {

        sameGroups.push_back(std::make_pair(i, j));
        meanAngleVec.push_back((lineGroupVec[i].mean + lineGroupVec[j].mean) /
                               2.0f);
      } else if ((lineGroupVec[i].mean < 10 ||
                  fabs(lineGroupVec[i].mean - 180) < 10) &&
                 fabs(fabs(lineGroupVec[i].mean - 180) - lineGroupVec[j].mean) <
                     10 &&
                 lineGroupVec[i].count + lineGroupVec[j].count != rows + cols) {

        sameGroups.push_back(std::make_pair(i, j));
        meanAngleVec.push_back(
            (fabs(lineGroupVec[i].mean - 180) + lineGroupVec[j].mean) / 2.0f);
      }
    }
  }

  std::vector<std::vector<int>> mergedIndices;
  std::vector<float> mergedAngles;

  if (sameGroups.size() > 0) {
    std::vector<int> member;
    member.push_back(sameGroups[0].first);
    member.push_back(sameGroups[0].second);
    mergedIndices.push_back(member);
    mergedAngles.push_back(meanAngleVec[0]);
    member.clear();

    if (sameGroups.size() > 1) {

      for (int i = 1; i < sameGroups.size(); i++) {

        int a = sameGroups[i].first;
        int b = sameGroups[i].second;

        bool aAbsent = true;
        bool bAbsent = true;

        for (int j = 0; j < mergedIndices.size(); j++) {
          bool aPresent = false;
          bool bPresent = false;

          for (int k = 0; k < mergedIndices[j].size(); k++) {

            if (a == mergedIndices[j][k]) {
              aPresent = true;
              aAbsent = false;
            }
            if (b == mergedIndices[j][k]) {
              bAbsent = false;
              bPresent = true;
            }
          }

          if (aPresent && bPresent) {
            // do nothing
          } else if (aPresent) {
            mergedIndices[j].push_back(b);
            mergedAngles[j] = (mergedAngles[j] + meanAngleVec[i]) / 2.0f;
          } else if (bPresent) {
            mergedIndices[j].push_back(a);
            mergedAngles[j] = (mergedAngles[j] + meanAngleVec[i]) / 2.0f;
          }
        }

        if (aAbsent && bAbsent) {
          std::vector<int> member2;
          member2.push_back(sameGroups[i].first);
          member2.push_back(sameGroups[i].second);
          mergedAngles.push_back(meanAngleVec[i]);
          mergedIndices.push_back(member2);
        }
      }
    }
  }

  lineGroupVecMerged.clear();

  // first lets merge the groups
  for (int i = 0; i < mergedIndices.size(); i++) {

    lineGroups member;
    std::vector<int> memberIndices;
    std::vector<int> memberLineLeft;
    std::vector<int> memberLineRight;
    member.count = 0;
    member.mean = mergedAngles[i];
    for (int j = 0; j < mergedIndices[i].size(); j++) {

      int index = mergedIndices[i][j];

      member.count += lineGroupVec[index].count;
      member.indices.insert(member.indices.end(),
                            lineGroupVec[index].indices.begin(),
                            lineGroupVec[index].indices.end());
      member.lineLeft.insert(member.lineLeft.end(),
                             lineGroupVec[index].lineLeft.begin(),
                             lineGroupVec[index].lineLeft.end());
      member.lineRight.insert(member.lineRight.end(),
                              lineGroupVec[index].lineRight.begin(),
                              lineGroupVec[index].lineRight.end());
    }
    lineGroupVecMerged.push_back(member);
  }

  // Now lets add all the ones that have been left out
  for (int m = 0; m < lineGroupVec.size(); m++) {

    bool present = false;
    for (int i = 0; i < mergedIndices.size(); i++) {

      for (int j = 0; j < mergedIndices[i].size(); j++) {

        if (m == mergedIndices[i][j])
          present = true;
      }
    }

    if (!present) {
      lineGroupVecMerged.push_back(lineGroupVec[m]);
    }
  }

  linesVec = lineGroupVecMerged;

  if (lineGroupVecMerged.size() < 2)
    return false;

  int constrainInd = 1000;
  int perpendicularInd = 1000;
  for (int i = 0; i < lineGroupVecMerged.size(); i++) {

    for (int j = 0; j < lineGroupVecMerged.size(); j++) {
      if (i == j)
        continue;

      if (fabs(fabs(lineGroupVecMerged[i].mean - lineGroupVecMerged[j].mean) -
               90) < angleDiff ||
          fabs(fabs(lineGroupVecMerged[i].mean - lineGroupVecMerged[j].mean) -
               270) < angleDiff) {
        if (lineGroupVecMerged[i].count == rows) {
          if (lineGroupVecMerged[j].count >= cols) {
            constrainInd = i;
            perpendicularInd = j;
          }
        } else if (lineGroupVecMerged[i].count == cols) {
          if (lineGroupVecMerged[j].count >= rows) {
            constrainInd = i;
            perpendicularInd = j;
          }
        }
      }
    }
  }

  if (constrainInd == 1000 || perpendicularInd == 1000) {
    return false;
  }

  int chosenInd = 1000;
  // find which side is varying
  if (lineGroupVecMerged[perpendicularInd].count == rows)
    chosenInd = constrainInd;
  else if (lineGroupVecMerged[perpendicularInd].count == cols)
    chosenInd = perpendicularInd;
  else if (lineGroupVecMerged[constrainInd].count == rows)
    chosenInd = perpendicularInd;
  else if (lineGroupVecMerged[constrainInd].count == cols)
    chosenInd = constrainInd;

  if (chosenInd == 1000)
    return false;

  std::vector<std::vector<int>> orderedPoints;
  std::vector<int> indiceVec;
  std::vector<int> yPts;

  for (int m = 0; m < lineGroupVecMerged[chosenInd].count; m++) {
    // Rotate points in all chosen indice
    int index = lineGroupVecMerged[chosenInd].indices[m];

    std::vector<int> xPts;
    std::vector<int> yVec;
    if (p_linePointCount[index] != rows)
      continue;

    for (int i = 0; i < p_linePointCount[index]; i++) {
      int2_t point = int2_t(p_linePoints[index * 40 + i] % resolution.x,
                            p_linePoints[index * 40 + i] / resolution.x);

      float angleDeg = lineGroupVecMerged[chosenInd].mean >= 180;
      if (angleDeg >= 180)
        angleDeg = angleDeg - 180;
      float angle = (-angleDeg * M_PI / 180.0f);

      int2_t rotatedPoint =
          rotatePoint(int2_t(resolution.x, resolution.y), -angle, point);

      xPts.push_back(rotatedPoint.x);
      yVec.push_back(rotatedPoint.y);
    }
    // Sorting line points by x
    std::vector<int> indicesPoints(xPts.size());
    std::iota(indicesPoints.begin(), indicesPoints.end(), 0);
    std::sort(indicesPoints.begin(), indicesPoints.end(),
              [&](int i, int j) { return (int)xPts[i] < xPts[j]; });

    yPts.push_back(yVec[indicesPoints[0]]);
    indiceVec.push_back(index);
    orderedPoints.push_back(indicesPoints);
  }

  // Sorting lines by y
  std::vector<int> indicesPointsY(yPts.size());
  std::iota(indicesPointsY.begin(), indicesPointsY.end(), 0);
  std::sort(indicesPointsY.begin(), indicesPointsY.end(),
            [&](int i, int j) { return (int)yPts[i] < yPts[j]; });

  for (int i = 0; i < indicesPointsY.size(); i++) {
    if (indicesPointsY.size() < cols ||
        orderedPoints[indicesPointsY[i]].size() < rows)
      return false;
  }

  // int none = 0;
  // int le
  // int right = 2;
  // int top = 3;
  // int bot = 4;

  if (indiceVec.size() < 1 || orderedPoints.size() < 1)
    return false;

  for (int i = 0; i < cols; i++) {
    int index = indiceVec[indicesPointsY[i]];

    for (int j = 0; j < rows; j++)
      imagePoints.push_back(p_linePoints[index * 40 + orderedPoints[i][j]]);
  }

  if (imagePoints.size() != rows * cols)
    return false;

  for (int i = 1; i < cols - 1; i++) {

    // find 2d distance
    int x0 = imagePoints[(i - 1) * rows] % resolution.x;
    int y0 = imagePoints[(i - 1) * rows] / resolution.x;

    int x1 = imagePoints[i * rows] % resolution.x;
    int y1 = imagePoints[i * rows] / resolution.x;

    int x2 = imagePoints[(i + 1) * rows] % resolution.x;
    int y2 = imagePoints[(i + 1) * rows] / resolution.x;

    float dist1 = getDist(x0, y0, x1, y1);
    float dist2 = getDist(x2, y2, x1, y1);

    float percentageDiff = fabsf(dist1 - dist2) / dist2;

    if (100 * percentageDiff > 80)
      return false;
  }

  return true;
}
