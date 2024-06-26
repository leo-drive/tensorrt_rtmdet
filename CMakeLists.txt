cmake_minimum_required(VERSION 3.8)
project(tensorrt_rtmdet)

find_package(tensorrt_common)
if(NOT ${tensorrt_common_FOUND})
  message(WARNING "The tensorrt_common package is not found. Please check its dependencies.")
  return()
endif()

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(OpenCV REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(WARNING "CUDA is not found. preprocess acceleration using CUDA will not be available.")
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# tensorrt_rtmdet
ament_auto_add_library(${PROJECT_NAME} SHARED
        src/tensorrt_rtmdet.cpp
)

ament_target_dependencies(${PROJECT_NAME}
        OpenCV
)

if(CMAKE_CUDA_COMPILER)
  # trt_batched_nms
  find_library(NVINFER_LIB nvinfer PATHS ${TENSORRT_ROOT}/lib)
  find_library(NVPLUGIN_LIB nvinfer_plugin PATHS ${TENSORRT_ROOT}/lib)

  include_directories(
          include
          include/tensorrt_rtmdet/trt_batched_nms
  )

  cuda_add_library(${PROJECT_NAME}_plugin SHARED
          src/trt_batched_nms/common_impl/nms/batched_nms_kernel.cpp
          src/trt_batched_nms/common_impl/nms/kernel.cu
          src/trt_batched_nms/common_impl/nms/allClassNMS.cu
          src/trt_batched_nms/common_impl/nms/gatherNMSOutputs.cu
          src/trt_batched_nms/common_impl/nms/permuteData.cu
          src/trt_batched_nms/common_impl/nms/sortScoresPerClass.cu
          src/trt_batched_nms/common_impl/nms/sortScoresPerImage.cu
          src/trt_batched_nms/batched_nms/trt_batched_nms.cpp
  )

  target_link_libraries(${PROJECT_NAME}_plugin ${CUDA_LIBRARIES} ${NVINFER_LIB} ${NVPLUGIN_LIB} ${OpenCV_LIBS} dl)

  # preprocess
  cuda_add_library(${PROJECT_NAME}_gpu_preprocess
          SHARED
          src/preprocess.cu
  )

  target_include_directories(${PROJECT_NAME}_gpu_preprocess PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  )

  # postprocess
  cuda_add_library(${PROJECT_NAME}_gpu_postprocess
          SHARED
          src/postprocess.cu
  )

  target_include_directories(${PROJECT_NAME}_gpu_postprocess PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  )

  target_link_libraries(${PROJECT_NAME}
          ${tensorrt_common_LIBRARIES}
          ${PROJECT_NAME}_gpu_preprocess
          ${PROJECT_NAME}_gpu_postprocess
          ${PROJECT_NAME}_plugin
  )

else()
  target_link_libraries(${PROJECT_NAME}
          ${tensorrt_common_LIBRARIES}
  )
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE
        TENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR}
)

# tensorrt_rtmdet_node
ament_auto_add_library(${PROJECT_NAME}_node SHARED
        src/tensorrt_rtmdet_node.cpp
)

ament_target_dependencies(${PROJECT_NAME}_node
        OpenCV
)

target_link_libraries(${PROJECT_NAME}_node
        ${PROJECT_NAME}
)

target_compile_definitions(${PROJECT_NAME}_node PRIVATE
        TENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR}
)

rclcpp_components_register_node(${PROJECT_NAME}_node
        PLUGIN "tensorrt_rtmdet::TrtRTMDetNode"
        EXECUTABLE tensorrt_rtmdet_node_exe
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE
        launch
        config
)