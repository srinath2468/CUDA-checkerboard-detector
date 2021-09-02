#include "main.hpp"

int main() {

  cv::Mat test = cv::imread("test2.jpg", 1);
  cv::cvtColor(test, test, CV_BGR2GRAY);

  checkerboard.Initialise(int2_t(test.cols, test.rows), int2_t(9, 6));

  checkerboard.Detect(test);

  checkerboard.Release();

  return 0;
}