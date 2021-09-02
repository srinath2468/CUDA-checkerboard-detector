#ifndef _REFINE_LINES_H_
#define _REFINE_LINES_H_

#include "common.h"

struct RefineLinesParams {

  int2_t resolution;
  int count;
  int rows;
  int cols;
};

void RefineLines(RefineLinesParams params, int *lines, int *linesPointCount,
                 int *linePoints, int *discardedLinePoints,
                 int *discardedLinePointCount);

#endif
