#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "getCheckerBoardLines.fatbin.c"
extern void __device_stub__Z14GetLinesKernel14GetLinesParamsPiS0_S0_S0_S0_(struct GetLinesParams&, int *, int *, int *, int *, int *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z14GetLinesKernel14GetLinesParamsPiS0_S0_S0_S0_(struct GetLinesParams&__par0, int *__par1, int *__par2, int *__par3, int *__par4, int *__par5){__cudaSetupArg(__par0, 0UL);__cudaSetupArgSimple(__par1, 24UL);__cudaSetupArgSimple(__par2, 32UL);__cudaSetupArgSimple(__par3, 40UL);__cudaSetupArgSimple(__par4, 48UL);__cudaSetupArgSimple(__par5, 56UL);__cudaLaunch(((char *)((void ( *)(struct GetLinesParams, int *, int *, int *, int *, int *))GetLinesKernel)));}
# 3 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/getCheckerBoardLines.cu"
void GetLinesKernel( struct GetLinesParams __cuda_0,int *__cuda_1,int *__cuda_2,int *__cuda_3,int *__cuda_4,int *__cuda_5)
# 5 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/getCheckerBoardLines.cu"
{__device_stub__Z14GetLinesKernel14GetLinesParamsPiS0_S0_S0_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 351 "/home/nvidia/checkerboard-detection/src/checkerboard/cuda/getCheckerBoardLines.cu"
}
# 1 "getCheckerBoardLines.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T7) {  __nv_dummy_param_ref(__T7); __nv_save_fatbinhandle_for_managed_rt(__T7); __cudaRegisterEntry(__T7, ((void ( *)(struct GetLinesParams, int *, int *, int *, int *, int *))GetLinesKernel), _Z14GetLinesKernel14GetLinesParamsPiS0_S0_S0_S0_, (-1)); __cudaRegisterVariable(__T7, __shadow_var(offsets,::offsets), 0, 72UL, 0, 0); __cudaRegisterVariable(__T7, __shadow_var(offsets7x7,::offsets7x7), 0, 392UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
