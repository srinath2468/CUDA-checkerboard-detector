#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "hessian.fatbin.c"
extern void __device_stub__Z14GaussianKernel13HessianParamsPhPf(struct HessianParams&, byte *, float *);
extern void __device_stub__Z26FirstOrderDerivativeKernel13HessianParamsPfS0_S0_S0_S0_S0_(struct HessianParams&, float *, float *, float *, float *, float *, float *);
extern void __device_stub__Z13HessianKernel13HessianParamsPfS0_S0_S0_S0_S0_S0_Pi(struct HessianParams&, float *, float *, float *, float *, float *, float *, float *, int *);
extern void __device_stub__Z15GetCornerKernel13HessianParamsPfS0_PiS1_S1_S1_(struct HessianParams&, float *, float *, int *, int *, int *, int *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z14GaussianKernel13HessianParamsPhPf(struct HessianParams&__par0, byte *__par1, float *__par2){__cudaSetupArg(__par0, 0UL);__cudaSetupArgSimple(__par1, 16UL);__cudaSetupArgSimple(__par2, 24UL);__cudaLaunch(((char *)((void ( *)(struct HessianParams, byte *, float *))GaussianKernel)));}
# 3 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
void GaussianKernel( struct HessianParams __cuda_0,byte *__cuda_1,float *__cuda_2)
# 4 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
{__device_stub__Z14GaussianKernel13HessianParamsPhPf( __cuda_0,__cuda_1,__cuda_2);
# 42 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
}
# 1 "hessian.cudafe1.stub.c"
void __device_stub__Z26FirstOrderDerivativeKernel13HessianParamsPfS0_S0_S0_S0_S0_( struct HessianParams&__par0,  float *__par1,  float *__par2,  float *__par3,  float *__par4,  float *__par5,  float *__par6) {  __cudaSetupArg(__par0, 0UL); __cudaSetupArgSimple(__par1, 16UL); __cudaSetupArgSimple(__par2, 24UL); __cudaSetupArgSimple(__par3, 32UL); __cudaSetupArgSimple(__par4, 40UL); __cudaSetupArgSimple(__par5, 48UL); __cudaSetupArgSimple(__par6, 56UL); __cudaLaunch(((char *)((void ( *)(struct HessianParams, float *, float *, float *, float *, float *, float *))FirstOrderDerivativeKernel))); }
# 44 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
void FirstOrderDerivativeKernel( struct HessianParams __cuda_0,float *__cuda_1,float *__cuda_2,float *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6)
# 47 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
{__device_stub__Z26FirstOrderDerivativeKernel13HessianParamsPfS0_S0_S0_S0_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 77 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
}
# 1 "hessian.cudafe1.stub.c"
void __device_stub__Z13HessianKernel13HessianParamsPfS0_S0_S0_S0_S0_S0_Pi( struct HessianParams&__par0,  float *__par1,  float *__par2,  float *__par3,  float *__par4,  float *__par5,  float *__par6,  float *__par7,  int *__par8) {  __cudaSetupArg(__par0, 0UL); __cudaSetupArgSimple(__par1, 16UL); __cudaSetupArgSimple(__par2, 24UL); __cudaSetupArgSimple(__par3, 32UL); __cudaSetupArgSimple(__par4, 40UL); __cudaSetupArgSimple(__par5, 48UL); __cudaSetupArgSimple(__par6, 56UL); __cudaSetupArgSimple(__par7, 64UL); __cudaSetupArgSimple(__par8, 72UL); __cudaLaunch(((char *)((void ( *)(struct HessianParams, float *, float *, float *, float *, float *, float *, float *, int *))HessianKernel))); }
# 79 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
void HessianKernel( struct HessianParams __cuda_0,float *__cuda_1,float *__cuda_2,float *__cuda_3,float *__cuda_4,float *__cuda_5,float *__cuda_6,float *__cuda_7,int *__cuda_8)
# 82 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
{__device_stub__Z13HessianKernel13HessianParamsPfS0_S0_S0_S0_S0_S0_Pi( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8);
# 132 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
}
# 1 "hessian.cudafe1.stub.c"
void __device_stub__Z15GetCornerKernel13HessianParamsPfS0_PiS1_S1_S1_( struct HessianParams&__par0,  float *__par1,  float *__par2,  int *__par3,  int *__par4,  int *__par5,  int *__par6) {  __cudaSetupArg(__par0, 0UL); __cudaSetupArgSimple(__par1, 16UL); __cudaSetupArgSimple(__par2, 24UL); __cudaSetupArgSimple(__par3, 32UL); __cudaSetupArgSimple(__par4, 40UL); __cudaSetupArgSimple(__par5, 48UL); __cudaSetupArgSimple(__par6, 56UL); __cudaLaunch(((char *)((void ( *)(struct HessianParams, float *, float *, int *, int *, int *, int *))GetCornerKernel))); }
# 134 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
void GetCornerKernel( struct HessianParams __cuda_0,float *__cuda_1,float *__cuda_2,int *__cuda_3,int *__cuda_4,int *__cuda_5,int *__cuda_6)
# 137 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
{__device_stub__Z15GetCornerKernel13HessianParamsPfS0_PiS1_S1_S1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 157 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/hessian.cu"
}
# 1 "hessian.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T2) {  __nv_dummy_param_ref(__T2); __nv_save_fatbinhandle_for_managed_rt(__T2); __cudaRegisterEntry(__T2, ((void ( *)(struct HessianParams, float *, float *, int *, int *, int *, int *))GetCornerKernel), _Z15GetCornerKernel13HessianParamsPfS0_PiS1_S1_S1_, (-1)); __cudaRegisterEntry(__T2, ((void ( *)(struct HessianParams, float *, float *, float *, float *, float *, float *, float *, int *))HessianKernel), _Z13HessianKernel13HessianParamsPfS0_S0_S0_S0_S0_S0_Pi, (-1)); __cudaRegisterEntry(__T2, ((void ( *)(struct HessianParams, float *, float *, float *, float *, float *, float *))FirstOrderDerivativeKernel), _Z26FirstOrderDerivativeKernel13HessianParamsPfS0_S0_S0_S0_S0_, (-1)); __cudaRegisterEntry(__T2, ((void ( *)(struct HessianParams, byte *, float *))GaussianKernel), _Z14GaussianKernel13HessianParamsPhPf, (-1)); __cudaRegisterVariable(__T2, __shadow_var(offsets,::offsets), 0, 72UL, 0, 0); __cudaRegisterVariable(__T2, __shadow_var(offsets7x7,::offsets7x7), 0, 392UL, 0, 0); __cudaRegisterVariable(__T2, __shadow_var(sobel7x7,::sobel7x7), 0, 392UL, 0, 0); __cudaRegisterVariable(__T2, __shadow_var(gaussianKernel,::gaussianKernel), 0, 36UL, 0, 0); __cudaRegisterVariable(__T2, __shadow_var(sobel,::sobel), 0, 72UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
