#include "checkerboard/checkerboard.hpp"
#include <opencv2/opencv.hpp>

namespace jcv {

bool Checkerboard::Initialise(int2_t resolution, int2_t boardSize) {

  this->_resolution = resolution;

  // Malloc Hessian Variables
  cudaMalloc((void **)&d_rgbImage,
             sizeof(unsigned char) * resolution.x * resolution.y);
  rgbCorners.InitialiseCorners(resolution);

  boardLengths = boardSize;

  lines.Initialise(resolution);

  // Initialising Cuda streams
  cudaStreamCreate(&stream);

  return true;
}

void Checkerboard::Detect(cv::Mat image) {

  cudaMemcpyAsync(d_rgbImage, image.data,
                  sizeof(unsigned char) *
                      (this->_resolution.x * this->_resolution.y),
                  cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  cv::Mat im2;
  cv::cvtColor(image, im2, CV_GRAY2BGR);

  rgbPoints.clear();
  std::vector<int> corners;

  if (rgbCorners.Detect(d_rgbImage, stream, corners)) {

    bool foundLines = lines.Detect(boardLengths, corners, stream);

    if (foundLines)
      rgbPoints.push_back(lines.ReturnPoints());

    if (foundLines) {

      for (int i = 0; i < rgbPoints[0].size(); i++) {
        cv::circle(im2, cv::Point(rgbPoints[0][i] % this->_resolution.x,
                                  rgbPoints[0][i] / this->_resolution.x),
                   2, cv::Scalar(0, 0, 255), 2);
      };

      cv::arrowedLine(
          im2, cv::Point(rgbPoints[0][0] % this->_resolution.x,
                         rgbPoints[0][0] / this->_resolution.x),
          cv::Point(rgbPoints[0][(boardLengths.x * boardLengths.y) - 1] %
                        this->_resolution.x,
                    rgbPoints[0][(boardLengths.x * boardLengths.y) - 1] /
                        this->_resolution.x),
          cv::Scalar(255, 0, 0), 2);
    }
  }

  cv::imwrite("im.jpg", im2);
}

void Checkerboard::Release() {

  // Destroy Stream
  cudaStreamDestroy(stream);

  // Release Corner and Line Detectors
  rgbCorners.Release();
  lines.Release();

  std::cout << "[Checkerboard] Released " << std::endl;
}
};
