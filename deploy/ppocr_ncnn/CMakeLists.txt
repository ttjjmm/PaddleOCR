cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 11)

project(ppocr)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#set(OpenCV_DIR "/usr/local/Cellar/opencv/4.3.0_5/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(ncnn_DIR "/home/tjm/opt/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
    message("ncnn FOUND")
endif()

#set(InferenceEngine_DIR "/opt/intel/openvino_2021/deployment_tools/inference_engine/share")
#find_package(InferenceEngine REQUIRED)
#if(InferenceEngine_FOUND)
#    message("InferenceEngine FOUND")
#endif()

#set(ngraph_DIR "/opt/intel/openvino_2021/deployment_tools/ngraph/cmake")
#find_package(ngraph REQUIRED)
#if(ngraph_FOUND)
#    message("ngraph FOUND")
#endif()



include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(ppocr main.cpp ppocr_det.cpp ppocr_det.h ppocr_cls.cpp
        ppocr_cls.h ppocr_rec.cpp ppocr_rec.h ppocr.cpp ppocr.h include/clipper.h)

target_link_libraries(
        ppocr
        ncnn
        #        ${InferenceEngine_LIBRARIES}
        #        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
)