#include "checkerboard/lines.h"

namespace jcv {

bool Lines::Initialise(int2_t resolution) {

  this->maxPoints = resolution.x * resolution.y;
  this->resolution = resolution;
  // Cuda Line variables
  cudaMalloc((void **)&d_lines, sizeof(int) * maxPoints * 2);
  cudaMalloc((void **)&d_linesCount, sizeof(int));
  cudaMalloc((void **)&d_linePoints, sizeof(int) * maxPoints * 40);
  cudaMalloc((void **)&d_linePointCount, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_cornerLocations, sizeof(int) * maxPoints);

  lines = (int *)malloc(sizeof(int) * maxPoints * 2);
  linesCount = (int *)malloc(sizeof(int));
  corners = (int *)malloc(sizeof(int) * maxPoints);
  linePointCount = (int *)malloc(sizeof(int) * maxPoints);
  linePoints = (int *)malloc(sizeof(int) * maxPoints * 40);
}

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

bool Lines::Detect(int2_t boardLengths, std::vector<int> &cornersVec,
                   cudaStream_t &stream) {

  this->cornersVec = cornersVec;
  this->boardLengths = boardLengths;
  this->stream = stream;

  points.clear();

  detecting = true;

  float angleDiff;
  if (side == 3 || side == 4)
    angleDiff = 20;
  else
    angleDiff = 20;

  int count = 0;
  for (int i = 0; i < cornersVec.size(); i++) {

    int x = cornersVec[i] % resolution.x;
    int y = cornersVec[i] / resolution.x;

    corners[count] = cornersVec[i];
    count++;
  }

  if (count < 5) {

    std::cout << " less corners " << std::endl;
    detecting = false;
    startDetection = false;

    return false;
  }

  getLinesParams.cornerCount = count;
  getLinesParams.resolution = resolution;
  getLinesParams.rows = boardLengths.x;
  getLinesParams.cols = boardLengths.y;

  cudaMemsetAsync(d_lines, 0, sizeof(int) * maxPoints * 2, stream);
  cudaMemsetAsync(d_linesCount, 0, sizeof(int), stream);

  if (side == 1 || side == 2) {
    getLinesParams.pixelDifference = 8;

  } else
    getLinesParams.pixelDifference = 6;

  cudaMemsetAsync(d_cornerLocations, 0, sizeof(int) * maxPoints, stream);
  cudaMemcpyAsync(d_cornerLocations, corners, sizeof(int) * count,
                  cudaMemcpyHostToDevice, stream);

  GetLines(getLinesParams, d_cornerLocations, d_linesCount, d_lines,
           d_linePoints, d_linePointCount, stream);

  cudaMemcpyAsync(linesCount, d_linesCount, sizeof(int), cudaMemcpyDeviceToHost,
                  stream);

  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(lines, d_lines, sizeof(int) * (*linesCount) * 2,
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(linePointCount, d_linePointCount, sizeof(int) * (*linesCount),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(linePoints, d_linePoints, sizeof(int) * (*linesCount) * 40,
                  cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  // Remove duplicate lines
  std::vector<int> linesPt1;
  std::vector<int> linesPt2;
  std::vector<int> lineIndices;

  std::cout << " number of lines " << *linesCount << std::endl;

  for (int i = 0; i < (*linesCount); i++) {
    int a = linePoints[i * 40];
    int b = linePoints[i * 40 + (linePointCount[i] - 1)];

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

      if (a / resolution.x == 0 || b / resolution.x == 0) {

        std::cout << "new mistake " << std::endl;
      }

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
                 lineGroupVec[i].count + lineGroupVec[j].count !=
                     boardLengths.x + boardLengths.y) {

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

        std::cout << " a " << a << " b " << b << std::endl;

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

  std::cout << " linegroup vec " << lineGroupVec.size() << std::endl;

  if (lineGroupVecMerged.size() < 2) {
    std::cout << " bad line merge " << std::endl;
    return false;
  }

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
        if (lineGroupVecMerged[i].count == boardLengths.x) {
          if (lineGroupVecMerged[j].count >= boardLengths.y) {
            constrainInd = i;
            perpendicularInd = j;
          }
        } else if (lineGroupVecMerged[i].count == boardLengths.y) {
          if (lineGroupVecMerged[j].count >= boardLengths.x) {
            constrainInd = i;
            perpendicularInd = j;
          }
        }
      }
    }
  }

  if (constrainInd == 1000 || perpendicularInd == 1000) {
    std::cout << " bad constrain or perp " << std::endl;
    return false;
  }

  int chosenInd = 1000;
  // find which side is varying
  if (lineGroupVecMerged[perpendicularInd].count == boardLengths.x)
    chosenInd = constrainInd;
  else if (lineGroupVecMerged[perpendicularInd].count == boardLengths.y)
    chosenInd = perpendicularInd;
  else if (lineGroupVecMerged[constrainInd].count == boardLengths.x)
    chosenInd = perpendicularInd;
  else if (lineGroupVecMerged[constrainInd].count == boardLengths.y)
    chosenInd = constrainInd;

  if (chosenInd == 1000) {

    std::cout << " bad chosen " << std::endl;
    return false;
  }

  std::vector<std::vector<int>> orderedPoints;
  std::vector<int> indiceVec;
  std::vector<int> yPts;

  for (int m = 0; m < lineGroupVecMerged[chosenInd].count; m++) {
    // Rotate points in all chosen indice
    int index = lineGroupVecMerged[chosenInd].indices[m];

    std::vector<int> xPts;
    std::vector<int> yVec;
    if (side == 3 || side == 4) {
      if (linePointCount[index] != boardLengths.x)
        continue;
    }

    for (int i = 0; i < linePointCount[index]; i++) {
      int2_t point = int2_t(linePoints[index * 40 + i] % resolution.x,
                            linePoints[index * 40 + i] / resolution.x);

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

  bool less = false;
  for (int i = 0; i < indicesPointsY.size(); i++) {
    if (indicesPointsY.size() < boardLengths.y ||
        orderedPoints[indicesPointsY[i]].size() < boardLengths.x) {
      less = true;
    }
  }

  std::cout << " ind " << indicesPointsY.size() << std::endl;

  if (less) {
    std::cout << " lesser than y " << std::endl;
    return false;
  }

  if (indiceVec.size() < 1 || orderedPoints.size() < 1) {
    return false;
  }

  std::vector<int> imagePoints;

  for (int i = 0; i < boardLengths.y; i++) {
    int index = indiceVec[indicesPointsY[i]];

    for (int j = 0; j < boardLengths.x; j++)
      imagePoints.push_back(linePoints[index * 40 + orderedPoints[i][j]]);
  }

  if (imagePoints.size() != boardLengths.y * boardLengths.x) {
    return false;
  }

  bool distBad = false;
  for (int i = 1; i < boardLengths.y - 1; i++) {

    // find 2d distance
    int x0 = imagePoints[(i - 1) * boardLengths.x] % resolution.x;
    int y0 = imagePoints[(i - 1) * boardLengths.x] / resolution.x;

    int x1 = imagePoints[i * boardLengths.x] % resolution.x;
    int y1 = imagePoints[i * boardLengths.x] / resolution.x;

    int x2 = imagePoints[(i + 1) * boardLengths.x] % resolution.x;
    int y2 = imagePoints[(i + 1) * boardLengths.x] / resolution.x;

    float dist1 = getDist(x0, y0, x1, y1);
    float dist2 = getDist(x2, y2, x1, y1);

    float percentageDiff = fabsf(dist1 - dist2) / dist2;

    if (100 * percentageDiff > 80) {
      distBad = true;
    }
  }

  // if (distBad) {
  //   std::cout << " bad dist " << std::endl;
  //   return false;
  // }

  points = imagePoints;
}

std::vector<int> Lines::ReturnPoints() { return points; }

bool Lines::DetectionStatus() { return detecting; }

void Lines::Release() {

  cudaFree(d_lines);
  cudaFree(d_linesCount);
  cudaFree(d_linePoints);
  cudaFree(d_linePointCount);
  cudaFree(d_cornerLocations);

  free(lines);
  free(linesCount);
  free(linePoints);
  free(linePointCount);
}
}