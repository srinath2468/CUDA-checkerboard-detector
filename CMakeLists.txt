cmake_minimum_required(VERSION 2.6)
project(CheckerboardLibrary)
set(CMAKE_CXX_STANDARD 11)

##---------------------------- SET THE CMAKE MODULE PATH ------------------------------------------------------------

#setting cmake module paths
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

##---------------------------------------------------------------------------------------------------------------------

include_directories(${CMAKE_SOURCE_DIR}/include ) #including the include directory to check for header files



set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_COMPILER g++)


find_package(boost QUIET)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

message("FOUND CUDA LIBRARIES:  ${CUDA_INCLUDE_DIRS}")


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}

    )
set(LIBRARIES
${CMAKE_DL_LIBS}
${OpenCV_LIBRARIES}
${Boost_LIBRARIES} 


)

##----------------SETTING HEADERS AND EXECUTABLES------------------------------------

# Source Files
set(SOURCE_FILES
src/main.cpp
src/checkerboard/checkerboard.cpp
src/checkerboard/lines.cpp
src/checkerboard/corners.cpp

)

#-Header Files
set(HEADER_FILES
include/main.hpp
include/checkerboard/checkerboard.hpp
include/checkerboard/lines.h
include/checkerboard/corners.h

)


#CUDA - Source Files
SET(CUDA_SOURCE_FILES
src/checkerboard/cuda/hessian.cu
src/checkerboard/cuda/corners.cu
src/checkerboard/cuda/getCheckerBoardLines.cu
)


#CUDA - Header Files
SET(CUDA_HEADER_FILES
include/checkerboard/cuda/hessian.h
include/checkerboard/cuda/cornersCuda.h
include/checkerboard/cuda/getLines.h
)

# -----------------------------------------------------------------------------------------------------------

set(CUDA_ARCH_BIN "61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-std=c++11;-lineinfo;-arch=sm_61")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false;--keep;--default-stream per-thread")

# -Xptxas -v
cuda_compile(CUDA_OBJS ${CUDA_SOURCE_FILES})


# --------------------------------------------------------------------------------------------------------------

##-------------------------- LIBRARY---------------------------------------------------------

Cuda_add_executable(checkerboard
	${SOURCE_FILES}
	${HEADER_FILES}
	${CUDA_SOURCE_FILES}
	${CUDA_HEADER_FILES}
)


#-------------------------- LINKING LIBRARY --------------------------------------------------


TARGET_LINK_LIBRARIES(checkerboard ${LIBRARIES})


#------------------------------------------------------------------------------------------------




