
#include "checkerboard/corners.h"

namespace jcv {

bool Corners::InitialiseCorners(int2_t resolution) {

  maxPoints = resolution.x * resolution.y;

  // Malloc Hessian Variables
  cudaMalloc((void **)&d_gaussian, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_xx, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_xy, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_yy, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_eigenValue1, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_eigenValue2, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_firstOrderDerivativeX, sizeof(float) * maxPoints);
  cudaMalloc((void **)&d_firstOrderDerivativeY, sizeof(float) * maxPoints);
  cudaMallocHost((void **)&p_gaussian, sizeof(float) * maxPoints);
  cudaMallocHost((void **)&p_bGaussian, sizeof(byte) * maxPoints);
  hessianParams.cornerThresh = 15;
  hessianParams.threads = maxPoints;
  hessianParams.resolution = resolution;

  // Malloc Filter Variables
  cudaMalloc((void **)&d_unfilteredCorners, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_maxCornerVal, sizeof(int));
  cudaMalloc((void **)&d_unfilteredCornerCount, sizeof(int));
  cudaMalloc((void **)&d_unfilteredCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_unfilteredCorner2d, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_filteredCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_filteredCornerCount, sizeof(int));
  cudaMallocHost((void **)&p_filteredCornerCount, sizeof(int));
  cudaMallocHost((void **)&p_unfilteredCornerCount, sizeof(int));
  cudaMallocHost((void **)&p_unfilteredCornerLocations,
                 sizeof(int) * maxPoints);
  filteredCornerLocations = (int *)malloc(sizeof(int) * maxPoints);

  // Malloc Fake Removal Variables
  cudaMalloc((void **)&d_realCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_realCornerCount, sizeof(int));
  cudaMallocHost((void **)&p_realCornerCount, sizeof(int));
  cudaMalloc((void **)&d_fakeCornerLocations, sizeof(int) * maxPoints);
  fakeCornerLocations = (int *)malloc(sizeof(int) * maxPoints);
  cornerParams.threads = maxPoints;
  cornerParams.resolution = resolution;

  // Malloc Erosion Variables
  cudaMalloc((void **)&d_erodedCornerLocations, sizeof(int) * maxPoints);
  cudaMalloc((void **)&d_erodedCornerCount, sizeof(int));
  cudaMallocHost((void **)&p_erodedCornerCount, sizeof(int));
  cornerLocations = (int *)malloc(sizeof(int) * maxPoints);

  return true;
}

bool Corners::Detect(byte *&image, cudaStream_t &stream,
                     std::vector<int> &corners) {

  cudaMemsetAsync(d_gaussian, 0, sizeof(float) * maxPoints, stream);
  cudaMemsetAsync(d_xx, 0, sizeof(float) * maxPoints, stream);
  cudaMemsetAsync(d_xy, 0, sizeof(float) * maxPoints, stream);
  cudaMemsetAsync(d_yy, 0, sizeof(float) * maxPoints, stream);
  cudaMemsetAsync(d_eigenValue1, 0, sizeof(float) * maxPoints, stream);
  cudaMemsetAsync(d_eigenValue2, 0, sizeof(float) * maxPoints, stream);
  cudaMemsetAsync(d_maxCornerVal, 0, sizeof(int), stream);
  cudaMemsetAsync(d_unfilteredCornerCount, 0, sizeof(int), stream);
  cudaMemsetAsync(d_unfilteredCorner2d, 0, sizeof(int) * maxPoints, stream);
  cudaMemsetAsync(d_filteredCornerCount, 0, sizeof(int), stream);
  cudaMemsetAsync(d_filteredCornerLocations, 0, sizeof(int) * maxPoints,
                  stream);

  Hessian(hessianParams, image, d_gaussian, d_xx, d_yy, d_xy,
          d_firstOrderDerivativeX, d_firstOrderDerivativeY, d_eigenValue1,
          d_eigenValue2, d_maxCornerVal, d_unfilteredCornerCount,
          d_unfilteredCornerLocations, d_unfilteredCorner2d, stream);

  cudaMemcpyAsync(p_unfilteredCornerCount, d_unfilteredCornerCount, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);

  cudaMemcpyAsync(p_gaussian, d_gaussian, sizeof(float) * maxPoints,
                  cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  float *testFloat = new float[640 * 480];
  unsigned char testByte[640 * 480] = {};

  for (int i = 0; i < 640 * 480; i++) {
    testByte[i] = (unsigned char)(int)p_gaussian[i];
  }

  cv::Mat testIm = cv::Mat(480, 640, CV_8UC1, testByte);

  cv::imwrite("gauss.jpg", testIm);

  FilterCorners(cornerParams, d_unfilteredCorner2d, d_filteredCornerLocations,
                d_filteredCornerCount, stream);

  cudaMemcpyAsync(p_filteredCornerCount, d_filteredCornerCount, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaMemcpyAsync(filteredCornerLocations, d_filteredCornerLocations,
                  sizeof(int) * (*p_filteredCornerCount),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cornerParams.count = *p_filteredCornerCount;

  std::vector<int> duplicateArray(filteredCornerLocations,
                                  filteredCornerLocations + cornerParams.count);

  std::sort(duplicateArray.begin(), duplicateArray.end());

  // Start traversing elements
  int n = cornerParams.count;

  if (n < 3)
    return false;

  int j = 0;
  int temp[n] = {};
  for (int i = 0; i < n - 1; i++) {

    if (duplicateArray[i] != duplicateArray[i + 1])
      fakeCornerLocations[j++] = duplicateArray[i];
  }
  fakeCornerLocations[j++] = duplicateArray[n - 1];

  cornerParams.fakeCornerCount = j;

  if (*p_unfilteredCornerCount < 10)
    return false;

  cudaMemsetAsync(d_realCornerCount, 0, sizeof(int), stream);
  cudaMemcpyAsync(d_fakeCornerLocations, fakeCornerLocations,
                  sizeof(int) * maxPoints, cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(d_realCornerLocations, 0, sizeof(int) * maxPoints, stream);

  RemoveFakeCorners(cornerParams, d_fakeCornerLocations, d_realCornerLocations,
                    d_realCornerCount, d_gaussian, stream);
  cudaMemsetAsync(d_filteredCornerLocations, 0, sizeof(int) * maxPoints,
                  stream);
  cudaMemsetAsync(d_filteredCornerCount, 0, sizeof(int), stream);
  FilterCorners(cornerParams, d_realCornerLocations, d_filteredCornerLocations,
                d_filteredCornerCount, stream);

  cudaMemcpyAsync(p_filteredCornerCount, d_filteredCornerCount, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(p_realCornerCount, d_realCornerCount, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaMemcpyAsync(filteredCornerLocations, d_filteredCornerLocations,
                  sizeof(int) * (*p_filteredCornerCount),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cornerParams.count = *p_realCornerCount;

  std::vector<int> duplicateArray2(
      filteredCornerLocations, filteredCornerLocations + cornerParams.count);

  std::sort(duplicateArray2.begin(), duplicateArray2.end());

  memset(fakeCornerLocations, 0, sizeof(int) * maxPoints);
  // Start traversing elements and remove duplicates
  int n2 = cornerParams.count;

  if (n2 < 3)
    return false;

  int j2 = 0;
  for (int i = 0; i < n2 - 1; i++) {

    if (duplicateArray2[i] != duplicateArray2[i + 1])
      fakeCornerLocations[j2++] = duplicateArray2[i];
  }
  fakeCornerLocations[j2++] = duplicateArray2[n2 - 1];

  cornerParams.realCornerCount = j2;

  cudaMemsetAsync(d_erodedCornerCount, 0, sizeof(int), stream);
  cudaMemsetAsync(d_erodedCornerLocations, 0, sizeof(int) * maxPoints, stream);

  cudaMemcpyAsync(d_realCornerLocations, fakeCornerLocations,
                  sizeof(int) * cornerParams.realCornerCount,
                  cudaMemcpyHostToDevice, stream);
  ErodedCorners(cornerParams, d_realCornerLocations, d_erodedCornerLocations,
                d_erodedCornerCount, stream);

  cudaMemcpyAsync(p_erodedCornerCount, d_erodedCornerCount, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  cudaMemcpyAsync(cornerLocations, d_erodedCornerLocations,
                  sizeof(int) * (*p_erodedCornerCount), cudaMemcpyDeviceToHost,
                  stream);

  cudaStreamSynchronize(stream);

  int cornerCount = *p_erodedCornerCount;

  for (int i = 0; i < cornerCount; i++) {

    corners.push_back(cornerLocations[i]);
  }
}

bool Corners::Release() {

  // Free Hessian
  cudaFree(d_gaussian);
  cudaFree(d_xx);
  cudaFree(d_xy);
  cudaFree(d_yy);
  cudaFree(d_eigenValue1);
  cudaFree(d_eigenValue2);
  cudaFree(d_firstOrderDerivativeX);
  cudaFree(d_firstOrderDerivativeY);
  cudaFreeHost(p_gaussian);
  cudaFreeHost(p_bGaussian);

  // Free Filter Variables
  cudaFree(d_unfilteredCorners);
  cudaFree(d_maxCornerVal);
  cudaFree(d_unfilteredCornerCount);
  cudaFree(d_unfilteredCornerLocations);
  cudaFree(d_unfilteredCorner2d);
  cudaFree(d_filteredCornerLocations);
  cudaFree(d_filteredCornerCount);
  cudaFreeHost(p_filteredCornerCount);
  cudaFreeHost(p_unfilteredCornerCount);
  cudaFreeHost(p_unfilteredCornerLocations);
  free(filteredCornerLocations);

  // Free Fake Removal Variables
  cudaFree(d_realCornerLocations);
  cudaFree(d_realCornerCount);
  cudaFree(d_fakeCornerLocations);
  cudaFreeHost(p_realCornerCount);
  free(fakeCornerLocations);

  // Free Erosion Variables
  cudaFree(d_erodedCornerCount);
  cudaFree(d_erodedCornerLocations);
  cudaFreeHost(p_erodedCornerCount);
  free(cornerLocations);
}
}