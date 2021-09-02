#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "corners.fatbin.c"
extern void __device_stub__Z18FilterCornerKernel12CornerParamsPiS0_S0_(struct CornerParams&, int *, int *, int *);
extern void __device_stub__Z22RemoveFakeCornerKernel12CornerParamsPiS0_S0_Pf(struct CornerParams&, int *, int *, int *, float *);
extern void __device_stub__Z18ErodedCornerKernel12CornerParamsPiS0_S0_(struct CornerParams&, int *, int *, int *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z18FilterCornerKernel12CornerParamsPiS0_S0_(struct CornerParams&__par0, int *__par1, int *__par2, int *__par3){__cudaSetupArg(__par0, 0UL);__cudaSetupArgSimple(__par1, 48UL);__cudaSetupArgSimple(__par2, 56UL);__cudaSetupArgSimple(__par3, 64UL);__cudaLaunch(((char *)((void ( *)(struct CornerParams, int *, int *, int *))FilterCornerKernel)));}
# 3 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
void FilterCornerKernel( struct CornerParams __cuda_0,int *__cuda_1,int *__cuda_2,int *__cuda_3)
# 5 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
{__device_stub__Z18FilterCornerKernel12CornerParamsPiS0_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 45 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
}
# 1 "corners.cudafe1.stub.c"
void __device_stub__Z22RemoveFakeCornerKernel12CornerParamsPiS0_S0_Pf( struct CornerParams&__par0,  int *__par1,  int *__par2,  int *__par3,  float *__par4) {  __cudaSetupArg(__par0, 0UL); __cudaSetupArgSimple(__par1, 48UL); __cudaSetupArgSimple(__par2, 56UL); __cudaSetupArgSimple(__par3, 64UL); __cudaSetupArgSimple(__par4, 72UL); __cudaLaunch(((char *)((void ( *)(struct CornerParams, int *, int *, int *, float *))RemoveFakeCornerKernel))); }
# 57 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
void RemoveFakeCornerKernel( struct CornerParams __cuda_0,int *__cuda_1,int *__cuda_2,int *__cuda_3,float *__cuda_4)
# 59 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
{__device_stub__Z22RemoveFakeCornerKernel12CornerParamsPiS0_S0_Pf( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 145 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
}
# 1 "corners.cudafe1.stub.c"
void __device_stub__Z18ErodedCornerKernel12CornerParamsPiS0_S0_( struct CornerParams&__par0,  int *__par1,  int *__par2,  int *__par3) {  __cudaSetupArg(__par0, 0UL); __cudaSetupArgSimple(__par1, 48UL); __cudaSetupArgSimple(__par2, 56UL); __cudaSetupArgSimple(__par3, 64UL); __cudaLaunch(((char *)((void ( *)(struct CornerParams, int *, int *, int *))ErodedCornerKernel))); }
# 163 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
void ErodedCornerKernel( struct CornerParams __cuda_0,int *__cuda_1,int *__cuda_2,int *__cuda_3)
# 164 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
{__device_stub__Z18ErodedCornerKernel12CornerParamsPiS0_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 190 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/corners.cu"
}
# 1 "corners.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T4) {  __nv_dummy_param_ref(__T4); __nv_save_fatbinhandle_for_managed_rt(__T4); __cudaRegisterEntry(__T4, ((void ( *)(struct CornerParams, int *, int *, int *))ErodedCornerKernel), _Z18ErodedCornerKernel12CornerParamsPiS0_S0_, (-1)); __cudaRegisterEntry(__T4, ((void ( *)(struct CornerParams, int *, int *, int *, float *))RemoveFakeCornerKernel), _Z22RemoveFakeCornerKernel12CornerParamsPiS0_S0_Pf, (-1)); __cudaRegisterEntry(__T4, ((void ( *)(struct CornerParams, int *, int *, int *))FilterCornerKernel), _Z18FilterCornerKernel12CornerParamsPiS0_S0_, (-1)); __cudaRegisterVariable(__T4, __shadow_var(offsets,::offsets), 0, 72UL, 0, 0); __cudaRegisterVariable(__T4, __shadow_var(offsets7x7,::offsets7x7), 0, 392UL, 0, 0); __cudaRegisterVariable(__T4, __shadow_var(centroOffsets,::centroOffsets), 0, 264UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
