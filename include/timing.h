
#include "common.h"

#include <chrono>
#include <iostream>
#include <string>
#include <time.h>
#ifndef _TIMING_H_
#define _TIMING_H_

// std::string RESET = "\033[0m";
// std::string BLACK = "\033[30m";              /* Black */
// std::string RED = "\033[31m";                /* Red */
// std::string GREEN = "\033[32m";              /* Green */
// std::string YELLOW = "\033[33m";             /* Yellow */
// std::string BLUE = "\033[34m";               /* Blue */
// std::string MAGENTA = "\033[35m";            /* Magenta */
// std::string CYAN = "\033[36m";               /* Cyan */
// std::string WHITE = "\033[37m";              /* White */
// std::string BOLDBLACK = "\033[1m\033[30m";   /* Bold Black */
// std::string BOLDRED = "\033[1m\033[31m";     /* Bold Red */
// std::string BOLDGREEN = "\033[1m\033[32m";   /* Bold Green */
// std::string BOLDYELLOW = "\033[1m\033[33m";  /* Bold Yellow */
// std::string BOLDBLUE = "\033[1m\033[34m";    /* Bold Blue */
// std::string BOLDMAGENTA = "\033[1m\033[35m"; /* Bold Magenta */
// std::string BOLDCYAN = "\033[1m\033[36m";    /* Bold Cyan */
// std::string BOLDWHITE = "\033[1m\033[37m";   /* Bold White */*

// std::chrono::system_clock::time_point start;

// void start_timer()
// {
//   start = std::chrono::system_clock::now();
// }

inline std::chrono::high_resolution_clock::time_point startTimer() {

  return std::chrono::high_resolution_clock::now();
}

inline void printTimer(std::chrono::high_resolution_clock::time_point time,
                       std::string c, int colour = 0) {
  std::string colStr = "";
  std::string resetStr = "\033[0m";
  if (colour == 3)
    colStr = "\033[35m"; // Magenta
  else if (colour == 2)
    colStr = "\033[37m"; // white
  else if (colour == 1)
    colStr = "\033[34m"; // blue
  else if (colour == 0)
    colStr = "\033[33m"; // yellow

  auto current = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(current - time)
          .count();

  std::cout << colStr << " Time taken for "
            << "" << c << " " << duration << " microseconds"
            << " or " << (1.0f / (float)(duration / 1000)) * 1000 << " fps "
            << resetStr << std::endl;
}

inline double returnTime(std::chrono::high_resolution_clock::time_point time) {
  auto current = std::chrono::high_resolution_clock::now();
  double duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(current - time)
          .count();

  return duration;
}

// void print_time(std::string c)
// {

//   std::chrono::system_clock::time_point b = std::chrono::system_clock::now();
//   std::chrono::duration<double> diff = b - start;

//   std::cout
//       << " Time taken for "
//       << "" << c << " "
//       << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
//       << " millisseconds"
//       << " or " << 1 / diff.count() << " fps " << std::endl;
// }

// void print_timer_ref(std::chrono::system_clock::time_point t, std::string
// c_ref)
// {
//   std::chrono::system_clock::time_point b_ref =
//       std::chrono::system_clock::now();
//   std::chrono::duration<double> diff = b_ref - t;
//   std::cout
//       << " {REF} Time taken for "
//       << "" << c_ref << " "
//       << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
//       << " millisseconds"
//       << " or " << 1 / diff.count() << " fps " << std::endl;
// }

// double returnTime(std::chrono::system_clock::time_point t)
// {
//   std::chrono::system_clock::time_point b_ref =
//       std::chrono::system_clock::now();
//   std::chrono::duration<double> timeDiff = b_ref - t;

//   double time =
//       std::chrono::duration_cast<std::chrono::milliseconds>(timeDiff).count();

//   return time;
// }

#endif
