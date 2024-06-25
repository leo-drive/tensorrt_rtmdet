//
// Created by bzeren on 14.06.2024.
//

#include "tensorrt_rtmdet/tensorrt_rtmdet_node.hpp"

#include <fstream>
#include <dlfcn.h>

void *loadLibrary(const char *libPath) {
    void *handle = dlopen(libPath, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot load library: " << dlerror() << '\n';
        return nullptr;
    }
    return handle;
}

namespace tensorrt_rtmdet {
    TrtRTMDetNode::TrtRTMDetNode(const rclcpp::NodeOptions &node_options)
            : Node("tensorrt_rtmdet", node_options) {
        RCLCPP_INFO(get_logger(), "tensorrt_rtmdet node has been started.");

        std::string onnxModel = "/home/bzeren/projects/labs/rtmdet/tensorrt_rtmdet_ws/onnx_model/end2end.onnx";
        std::string engineFile = "/home/bzeren/projects/labs/rtmdet/tensorrt_rtmdet_ws/tensorrt_model/end2end.engine";
        std::string pluginFile = "/home/bzeren/projects/labs/rtmdet/tensorrt_rtmdet_ws/build/tensorrt_rtmdet/libtensorrt_rtmdet_plugin.so";
        std::string videoFile = "/home/bzeren/projects/labs/rtmdet/road.mp4";
        std::string outputVideoFile = "/home/bzeren/projects/labs/rtmdet/tensorrt_rtmdet_ws/output.mp4";
        std::string precision = "fp16";

        if (!loadLibrary(pluginFile.c_str())) {
            std::cerr << "Error when loading plugin" << std::endl;
        }

        tensorrt_common::BuildConfig build_config(
                "MinMax", -1, false, false, false, 6.0
        );

        const double norm_factor = 1.0;
        const std::string cache_dir = "/tmp/rtmdet";
        const tensorrt_common::BatchConfig batch_config{1, 1, 1};
        const size_t max_workspace_size = (1 << 30);

        trt_rtmdet_ = std::make_unique<tensorrt_rtmdet::TrtRTMDet>(
                onnxModel, precision, 80, 0.3, 0.3, build_config,
                false, "", norm_factor, cache_dir, batch_config, max_workspace_size, ""
        );

        cv::VideoCapture cap(videoFile);

        while (rclcpp::ok()) {
            cv::Mat frame;
            // Capture frame-by-frame
            cap >> frame;

            if (frame.empty())
                break;

            tensorrt_rtmdet::ObjectArrays objects;

            std::cout << "Start inference" << std::endl;
            if (!trt_rtmdet_->doInference({frame}, objects)) {
                RCLCPP_WARN(this->get_logger(), "Fail to inference");
                return;
            }
            std::cout << "End inference" << std::endl;
        }

    }
} // namespace tensorrt_rtmdet

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(tensorrt_rtmdet::TrtRTMDetNode)