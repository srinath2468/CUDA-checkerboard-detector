#ifndef _COMMON_H
#define _COMMON_H

#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <math.h>
#include <opencv2/opencv.hpp>

typedef unsigned char byte;

__device__ const int offsets[9][2] = {
    {-1, -1}, // 0
    {0, -1},  // 1
    {1, -1},  // 2
    {-1, 0},  // 3
    {0, 0},   // 4
    {1, 0},   // 5
    {-1, 1},  // 6
    {0, 1},   // 7
    {1, 1}    // 8
};

__device__ const int offsets7x7[49][2] = {
    {-3, -3}, // 0
    {-2, -3}, // 1
    {1, -3},  // 2
    {0, -3},  // 3
    {1, -3},  // 4
    {2, -3},  // 5
    {3, -3},  // 6

    {-3, -2}, // 0
    {-2, -2}, // 1
    {1, -2},  // 2
    {0, -2},  // 3
    {1, -2},  // 4
    {2, -2},  // 5
    {3, -2},  // 6

    {-3, -1}, // 0
    {-2, -1}, // 1
    {1, -1},  // 2
    {0, -1},  // 3
    {1, -1},  // 4
    {2, -1},  // 5
    {3, -1},  // 6

    {-3, 0}, // 0
    {-2, 0}, // 1
    {1, 0},  // 2
    {0, 0},  // 3
    {1, 0},  // 4
    {2, 0},  // 5
    {3, 0},  // 6

    {-3, 1}, // 0
    {-2, 1}, // 1
    {1, 1},  // 2
    {0, 1},  // 3
    {1, 1},  // 4
    {2, 1},  // 5
    {3, 1},  // 6

    {-3, 2}, // 0
    {-2, 2}, // 1
    {1, 2},  // 2
    {0, 2},  // 3
    {1, 2},  // 4
    {2, 2},  // 5
    {3, 2},  // 6

    {-3, 3}, // 0
    {-2, 3}, // 1
    {1, 3},  // 2
    {0, 3},  // 3
    {1, 3},  // 4
    {2, 3},  // 5
    {3, 3}   // 6

};

struct byte3_t {
  byte3_t() {}
  byte3_t(byte x, byte y, byte z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  byte x;
  byte y;
  byte z;
};

struct int2_t {
  int2_t() {}
  int2_t(int x, int y) {
    this->x = x;
    this->y = y;
  }

  int x;
  int y;
};

struct int3_t {
  int3_t() {}
  int3_t(int x, int y, int z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  int x;
  int y;
  int z;
};

struct float2_t {
  float2_t() {}
  float2_t(float x, float y) {
    this->x = x;
    this->y = y;
  }

  float x;
  float y;
};

struct float3_t {
  float3_t() {}
  float3_t(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  float x;
  float y;
  float z;
};

struct float4_t {
  float4_t() {}
  float4_t(float x, float y, float z, float w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
  }

  float x;
  float y;
  float z;
  float w;
};

struct string3_t {
  string3_t() {}
  string3_t(std::string x, std::string y, std::string z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  std::string x;
  std::string y;
  std::string z;
};

struct string4_t {
  string4_t() {}
  string4_t(std::string x, std::string y, std::string z, std::string w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
  }

  std::string x;
  std::string y;
  std::string z;
  std::string w;
};

struct matrix4x4_t {
  matrix4x4_t() {
    this->rows[0] = float4_t(1, 0, 0, 0);
    this->rows[1] = float4_t(0, 1, 0, 0);
    this->rows[2] = float4_t(0, 0, 1, 0);
    this->rows[3] = float4_t(0, 0, 0, 1);
  }
  matrix4x4_t(float4_t row1, float4_t row2, float4_t row3, float4_t row4) {
    this->rows[0] = row1;
    this->rows[1] = row2;
    this->rows[2] = row3;
    this->rows[3] = row4;
  }

  float4_t rows[4];
};

struct CameraTransform {

  // Position
  float x = 0;
  float y = 0;
  float z = 0;

  // Quaternion
  float qw = 1;
  float qx = 0;
  float qy = 0;
  float qz = 0;
};

struct Intrinsics {
  Intrinsics() {}
  Intrinsics(std::string model, float2_t c, float2_t f, float k0, float k1,
             float k2, float k3, float k4) {
    this->model = model;
    this->c = c;
    this->f = f;
    this->k0 = k0;
    this->k1 = k1;
    this->k2 = k2;
    this->k3 = k3;
    this->k4 = k4;
  }

  std::string model;
  float2_t c;
  float2_t f;

  float k0;
  float k1;
  float k2;
  float k3;
  float k4;
};

inline float distance3_f(float x1, float y1, float z1, float x2, float y2,
                         float z2) {

  float d = sqrtf(powf((x1 - x2), 2) + powf((y1 - y2), 2) + powf((z1 - z2), 2));
  return d;
}

#endif
